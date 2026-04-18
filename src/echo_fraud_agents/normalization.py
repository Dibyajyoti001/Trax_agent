from __future__ import annotations

import csv
import io
import json
import re
from email import policy
from email.parser import BytesParser
from pathlib import PurePosixPath

from echo_fraud_agents.models import (
    AudioAsset,
    DatasetManifest,
    LocationEvent,
    MessageEvent,
    NormalizedDataset,
    TransactionRecord,
    UserProfile,
)
from echo_fraud_agents.utils import (
    compact_text,
    extract_domain,
    extract_urls,
    file_basename,
    normalize_text,
    parse_timestamp,
    safe_float,
    safe_int,
    slugify,
    strip_html,
)


SMS_FIELD_RE = re.compile(r"^(From|To|Date|Message):\s*(.*)$", re.IGNORECASE)


def normalize_dataset(bundle) -> NormalizedDataset:
    transactions = _load_transactions(bundle.files)
    users = _load_users(bundle.files)
    messages = _load_messages(bundle.files)
    audio_assets = _load_audio_assets(bundle.files)
    locations = _load_locations(bundle.files)

    modalities = ["transactions", "users"]
    if messages:
        modalities.append("messages")
    if locations:
        modalities.append("locations")
    if audio_assets:
        modalities.append("audio")

    manifest = DatasetManifest(
        name=bundle.name,
        slug=slugify(bundle.name),
        source_label=bundle.source_label,
        modalities=modalities,
        record_counts={
            "transactions": len(transactions),
            "users": len(users),
            "messages": len(messages),
            "audio_assets": len(audio_assets),
            "locations": len(locations),
        },
        notes=[
            f"bundle_source={bundle.source_label}",
            f"file_count={len(bundle.files)}",
        ],
    )

    schema = {
        "files": sorted(bundle.files),
        "transactions_columns": sorted({key for item in transactions for key in item.raw}),
        "user_fields": sorted({key for item in users for key in item.raw}),
        "message_channels": sorted({item.channel for item in messages}),
        "location_fields": sorted({key for item in locations for key in item.raw}),
    }

    return NormalizedDataset(
        bundle=bundle,
        manifest=manifest,
        schema=schema,
        transactions=transactions,
        users=users,
        messages=messages,
        audio_assets=audio_assets,
        locations=locations,
    )


def _load_transactions(files: dict[str, bytes]) -> list[TransactionRecord]:
    target = next((path for path in files if path.lower().endswith("transactions.csv")), None)
    if not target:
        return []
    records: list[TransactionRecord] = []
    reader = csv.DictReader(io.TextIOWrapper(io.BytesIO(files[target]), encoding="utf-8"))
    for row in reader:
        description = (row.get("description") or "").strip()
        location_label = (row.get("location") or "").strip()
        merchant_hint = normalize_text(description or (location_label.split(" - ")[-1] if location_label else ""))
        records.append(
            TransactionRecord(
                transaction_id=(row.get("transaction_id") or "").strip(),
                sender_id=(row.get("sender_id") or "").strip(),
                recipient_id=(row.get("recipient_id") or "").strip(),
                transaction_type=(row.get("transaction_type") or "").strip(),
                amount=safe_float(row.get("amount")),
                location_label=location_label,
                payment_method=(row.get("payment_method") or "").strip(),
                sender_iban=(row.get("sender_iban") or "").strip(),
                recipient_iban=(row.get("recipient_iban") or "").strip(),
                balance_after=safe_float(row.get("balance_after")),
                description=description,
                timestamp=parse_timestamp(row.get("timestamp")),
                merchant_hint=merchant_hint,
                raw=dict(row),
            )
        )
    return records


def _load_users(files: dict[str, bytes]) -> list[UserProfile]:
    target = next((path for path in files if path.lower().endswith("users.json")), None)
    if not target:
        return []
    payload = json.loads(files[target].decode("utf-8"))
    users: list[UserProfile] = []
    for row in payload:
        first = str(row.get("first_name") or "").strip()
        last = str(row.get("last_name") or "").strip()
        residence = row.get("residence") or {}
        full_name = f"{first} {last}".strip()
        users.append(
            UserProfile(
                owner_key=normalize_text(full_name),
                first_name=first,
                last_name=last,
                full_name=full_name,
                birth_year=safe_int(row.get("birth_year"), default=0) or None,
                salary=safe_float(row.get("salary")),
                job=str(row.get("job") or "").strip(),
                iban=str(row.get("iban") or "").strip(),
                residence_city=str(residence.get("city") or "").strip(),
                residence_lat=safe_float(residence.get("lat")),
                residence_lng=safe_float(residence.get("lng")),
                description=compact_text(str(row.get("description") or ""), 900),
                raw=dict(row),
            )
        )
    return users


def _load_messages(files: dict[str, bytes]) -> list[MessageEvent]:
    messages: list[MessageEvent] = []
    sms_target = next((path for path in files if path.lower().endswith("sms.json")), None)
    mail_target = next((path for path in files if path.lower().endswith("mails.json")), None)
    if sms_target:
        payload = json.loads(files[sms_target].decode("utf-8"))
        for index, row in enumerate(payload):
            messages.append(_normalize_sms(row.get("sms") or "", index))
    if mail_target:
        payload = json.loads(files[mail_target].decode("utf-8"))
        for index, row in enumerate(payload):
            messages.append(_normalize_mail(row.get("mail") or "", index))
    return messages


def _normalize_sms(raw_text: str, index: int) -> MessageEvent:
    parsed: dict[str, str] = {}
    message_lines: list[str] = []
    for line in raw_text.splitlines():
        match = SMS_FIELD_RE.match(line)
        if match and match.group(1).lower() != "message":
            parsed[match.group(1).lower()] = match.group(2).strip()
            continue
        if match and match.group(1).lower() == "message":
            message_lines.append(match.group(2).strip())
            continue
        if message_lines:
            message_lines.append(line.strip())
    body = " ".join(part for part in message_lines if part)
    return MessageEvent(
        message_id=f"sms-{index}",
        channel="sms",
        sender_label=parsed.get("from", ""),
        sender_domain=extract_domain(parsed.get("from", "")),
        recipient_hint=parsed.get("to", ""),
        owner_key=None,
        timestamp=parse_timestamp(parsed.get("date")),
        subject="",
        body_text=body,
        body_preview=compact_text(body, 260),
        urls=extract_urls(body),
        raw_text=raw_text,
    )


def _normalize_mail(raw_text: str, index: int) -> MessageEvent:
    message = BytesParser(policy=policy.default).parsebytes(raw_text.encode("utf-8", errors="ignore"))
    sender = message.get("From", "")
    recipient = message.get("To", "")
    subject = message.get("Subject", "")
    date_value = message.get("Date", "")
    if message.is_multipart():
        body_parts = []
        for part in message.walk():
            if part.get_content_maintype() == "multipart":
                continue
            content = part.get_content()
            if part.get_content_subtype() == "html":
                body_parts.append(strip_html(content))
            else:
                body_parts.append(str(content))
        body_text = " ".join(part.strip() for part in body_parts if part)
    else:
        content = message.get_content()
        body_text = strip_html(content) if "<html" in str(content).lower() else str(content)
    return MessageEvent(
        message_id=f"mail-{index}",
        channel="mail",
        sender_label=sender,
        sender_domain=extract_domain(sender),
        recipient_hint=recipient,
        owner_key=None,
        timestamp=parse_timestamp(date_value),
        subject=compact_text(subject, 180),
        body_text=compact_text(body_text, 5000),
        body_preview=compact_text(f"{subject} {body_text}", 260),
        urls=extract_urls(raw_text),
        raw_text=raw_text,
    )


def _load_audio_assets(files: dict[str, bytes]) -> list[AudioAsset]:
    supported = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    assets: list[AudioAsset] = []
    for relative_path, payload in files.items():
        suffix = PurePosixPath(relative_path).suffix.lower()
        if suffix not in supported:
            continue
        basename = file_basename(relative_path)
        stem = PurePosixPath(basename).stem
        owner_hint = ""
        timestamp = None
        if "-" in stem:
            ts_text, _, name_part = stem.partition("-")
            timestamp = parse_timestamp(ts_text)
            owner_hint = normalize_text(name_part.replace("_", " "))
        assets.append(
            AudioAsset(
                asset_id=normalize_text(f"{relative_path}-{len(payload)}"),
                filename=basename,
                relative_path=relative_path,
                owner_key=None,
                owner_hint=owner_hint,
                timestamp=timestamp,
                extension=suffix.lstrip("."),
                size_bytes=len(payload),
            )
        )
    return assets


def _load_locations(files: dict[str, bytes]) -> list[LocationEvent]:
    target = next((path for path in files if path.lower().endswith("locations.json")), None)
    if not target:
        return []
    payload = json.loads(files[target].decode("utf-8"))
    events: list[LocationEvent] = []
    for index, row in enumerate(payload):
        events.append(
            LocationEvent(
                event_id=f"loc-{index}",
                biotag=str(row.get("biotag") or "").strip(),
                owner_key=None,
                city=str(row.get("city") or "").strip(),
                lat=safe_float(row.get("lat")),
                lng=safe_float(row.get("lng")),
                timestamp=parse_timestamp(row.get("timestamp")),
                raw=dict(row),
            )
        )
    return events
