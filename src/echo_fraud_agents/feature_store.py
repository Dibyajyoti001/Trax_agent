from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from echo_fraud_agents.config import AppConfig
from echo_fraud_agents.graph_index import GraphIndex
from echo_fraud_agents.models import (
    AudioAsset,
    CandidateSeed,
    CaseCandidate,
    MessageEvent,
    NormalizedDataset,
    TransactionRecord,
    UserProfile,
)
from echo_fraud_agents.utils import (
    compact_text,
    entity_signature,
    extract_domain,
    mean,
    name_tokens,
    normalize_text,
    pstdev,
    quantile,
)


KEYWORD_GROUPS = {
    "urgency": {"urgent", "immediately", "right away", "now", "asap", "final notice"},
    "verification": {"verify", "validation", "confirm", "update account", "secure account"},
    "credentials": {"password", "otp", "code", "credential", "login", "pin"},
    "payments": {"transfer", "wire", "payment", "invoice", "refund", "gift card", "crypto"},
    "coercion": {"suspended", "locked", "blocked", "failure", "penalty", "legal"},
}


@dataclass(slots=True)
class FeatureStore:
    dataset: NormalizedDataset
    users_by_owner: dict[str, UserProfile]
    users_by_iban: dict[str, UserProfile]
    transactions: list[TransactionRecord]
    transactions_by_id: dict[str, TransactionRecord]
    messages: list[MessageEvent]
    messages_by_owner: dict[str, list[MessageEvent]]
    audio_assets: list[AudioAsset]
    audio_by_owner: dict[str, list[AudioAsset]]
    locations_by_owner: dict[str, list[dict[str, Any]]]
    owner_aliases: dict[str, set[str]]
    sender_histories: dict[str, list[TransactionRecord]]
    recipient_histories: dict[str, list[TransactionRecord]]
    owner_histories: dict[str, list[TransactionRecord]]
    pair_histories: dict[tuple[str, str], list[TransactionRecord]]
    dataset_stats: dict[str, Any]
    graph: GraphIndex = field(repr=False)

    def linked_messages_for(
        self,
        tx: TransactionRecord,
        *,
        lookback_hours: int,
        max_items: int | None = None,
    ) -> list[MessageEvent]:
        events = self.messages_by_owner.get(tx.owner_key or "", [])
        linked = [
            event
            for event in events
            if _within_hours(tx.timestamp, event.timestamp, lookback_hours)
        ]
        linked.sort(key=lambda event: event.risk_score, reverse=True)
        return linked[:max_items] if max_items else linked

    def linked_audio_for(
        self,
        tx: TransactionRecord,
        *,
        lookback_hours: int,
        max_items: int | None = None,
    ) -> list[AudioAsset]:
        assets = self.audio_by_owner.get(tx.owner_key or "", [])
        linked = [
            asset
            for asset in assets
            if _within_hours(tx.timestamp, asset.timestamp, lookback_hours)
        ]
        linked.sort(key=lambda asset: _timestamp_sort_value(asset.timestamp), reverse=True)
        return linked[:max_items] if max_items else linked

    def transaction_summary(self, tx: TransactionRecord, candidate_score: float | None = None) -> dict[str, Any]:
        user = self.users_by_owner.get(tx.owner_key or "")
        return {
            "transaction_id": tx.transaction_id,
            "owner_key": tx.owner_key,
            "direction": tx.direction,
            "sender_id": tx.sender_id,
            "recipient_id": tx.recipient_id,
            "amount": tx.amount,
            "transaction_type": tx.transaction_type,
            "payment_method": tx.payment_method,
            "location": tx.location_label,
            "description": compact_text(tx.description, 160),
            "timestamp": tx.timestamp.isoformat() if tx.timestamp else None,
            "candidate_score": candidate_score,
            "economic_severity": self.economic_severity(tx),
            "owner_profile": {
                "salary": user.salary if user else 0.0,
                "job": user.job if user else "",
                "residence_city": user.residence_city if user else "",
                "birth_year": user.birth_year if user else None,
            },
        }

    def transaction_pattern_summary(self, tx: TransactionRecord) -> dict[str, Any]:
        prior = _prior_records(self.owner_histories.get(tx.owner_key or "", []), tx.timestamp, tx.transaction_id)
        sender_prior = _prior_records(self.sender_histories.get(tx.sender_id, []), tx.timestamp, tx.transaction_id)
        amounts = [item.amount for item in sender_prior]
        recipients = {item.recipient_id for item in sender_prior if item.recipient_id}
        pair_count = len(self.pair_histories.get((tx.sender_id, tx.recipient_id), []))
        payment_counts = Counter(item.payment_method for item in prior if item.payment_method)
        recent_30m = _count_within_hours(prior, tx.timestamp, 0.5)
        recent_3h = _count_within_hours(prior, tx.timestamp, 3)
        return {
            "prior_count": len(sender_prior),
            "pair_prior_count": pair_count,
            "mean_amount": mean(amounts),
            "p90_amount": quantile(amounts, 0.90),
            "amount_std": pstdev(amounts),
            "known_recipients": len(recipients),
            "recent_30m_count": recent_30m,
            "recent_3h_count": recent_3h,
            "payment_method_usage": payment_counts.most_common(4),
            "current_payment_method_prior": payment_counts.get(tx.payment_method, 0),
        }

    def geo_behavior_summary(self, tx: TransactionRecord) -> dict[str, Any]:
        user = self.users_by_owner.get(tx.owner_key or "")
        prior = _prior_records(self.owner_histories.get(tx.owner_key or "", []), tx.timestamp, tx.transaction_id)
        recent_locations = self.locations_by_owner.get(tx.owner_key or "", [])
        recent_cities = [item["city"] for item in recent_locations[-12:]]
        hours = [item.timestamp.hour for item in prior if item.timestamp]
        hour_count = Counter(hours)
        tx_hour = tx.timestamp.hour if tx.timestamp else None
        tx_city = _transaction_city(tx.location_label)
        return {
            "residence_city": user.residence_city if user else "",
            "transaction_city": tx_city,
            "recent_location_cities": recent_cities[-6:],
            "known_city_count": len(set(recent_cities)),
            "hour_histogram": hour_count.most_common(6),
            "current_hour_seen": hour_count.get(tx_hour, 0) if tx_hour is not None else 0,
            "location_novelty": float(tx_city not in {normalize_text(city) for city in recent_cities if city}),
        }

    def graph_summary(self, tx: TransactionRecord) -> dict[str, Any]:
        return self.graph.summary_for(tx.transaction_id)

    def economic_severity(self, tx: TransactionRecord) -> float:
        user = self.users_by_owner.get(tx.owner_key or "")
        salary = user.salary if user else 0.0
        monthly = salary / 12 if salary else 0.0
        if monthly <= 0:
            return min(1.0, tx.amount / max(2000.0, self.dataset_stats["high_value_amount"]))
        return min(1.0, tx.amount / max(monthly, 1.0))

    def transactions_for_ids(self, transaction_ids: list[str]) -> list[TransactionRecord]:
        items = [self.transactions_by_id[tx_id] for tx_id in transaction_ids if tx_id in self.transactions_by_id]
        items.sort(key=lambda item: item.timestamp or datetime.min)
        return items

    def case_context(
        self,
        case: CaseCandidate,
        seeds: dict[str, CandidateSeed],
        *,
        max_transaction_items: int,
        max_message_items: int,
        max_audio_items: int,
        message_lookback_hours: int,
        audio_lookback_hours: int,
    ) -> dict[str, Any]:
        transactions = self.transactions_for_ids(case.member_transaction_ids)
        anchors = self.transactions_for_ids(case.anchor_transaction_ids)
        linked_messages = self._case_messages(
            transactions,
            lookback_hours=message_lookback_hours,
            max_items=max_message_items,
        )
        linked_audio = self._case_audio(
            transactions,
            lookback_hours=audio_lookback_hours,
            max_items=max_audio_items,
        )
        owner_profiles = []
        for owner_key in case.owner_keys:
            user = self.users_by_owner.get(owner_key)
            if user is None:
                continue
            owner_profiles.append(
                {
                    "owner_key": owner_key,
                    "full_name": user.full_name,
                    "job": user.job,
                    "salary": user.salary,
                    "residence_city": user.residence_city,
                }
            )
        return {
            "case_id": case.case_id,
            "case_type": case.case_type,
            "priority_hint": case.priority_hint,
            "economic_severity": case.economic_severity,
            "anchor_transaction_ids": case.anchor_transaction_ids,
            "member_transaction_ids": case.member_transaction_ids,
            "seed_transaction_ids": case.seed_transaction_ids,
            "shared_entities": case.shared_entities,
            "modality_coverage": case.modality_coverage,
            "summary": case.summary,
            "owner_profiles": owner_profiles,
            "anchor_transactions": [
                self.transaction_summary(tx, seeds.get(tx.transaction_id).score if tx.transaction_id in seeds else None)
                for tx in anchors[:max_transaction_items]
            ],
            "member_transactions": [
                self.transaction_summary(tx, seeds.get(tx.transaction_id).score if tx.transaction_id in seeds else None)
                for tx in transactions[:max_transaction_items]
            ],
            "transaction_pattern_summary": self.case_pattern_summary(transactions),
            "geo_behavior_summary": self.case_geo_summary(transactions),
            "graph_summary": self.case_graph_summary(case.member_transaction_ids),
            "linked_messages": [
                {
                    "message_id": item.message_id,
                    "channel": item.channel,
                    "sender": item.sender_label,
                    "sender_domain": item.sender_domain,
                    "subject": compact_text(item.subject, 120),
                    "preview": item.body_preview,
                    "risk_score": item.risk_score,
                    "risk_reasons": item.risk_reasons,
                    "urls": item.urls[:4],
                    "timestamp": item.timestamp.isoformat() if item.timestamp else None,
                }
                for item in linked_messages
            ],
            "linked_audio": [
                {
                    "asset_id": item.asset_id,
                    "filename": item.filename,
                    "owner_key": item.owner_key,
                    "timestamp": item.timestamp.isoformat() if item.timestamp else None,
                    "size_bytes": item.size_bytes,
                    "extension": item.extension,
                }
                for item in linked_audio
            ],
            "seed_scores": {
                tx_id: {
                    "score": seeds[tx_id].score,
                    "economic_severity": seeds[tx_id].economic_severity,
                    "reasons": seeds[tx_id].reasons,
                    "signal_breakdown": seeds[tx_id].signal_breakdown,
                }
                for tx_id in case.seed_transaction_ids
                if tx_id in seeds
            },
        }

    def case_pattern_summary(self, transactions: list[TransactionRecord]) -> dict[str, Any]:
        amounts = [tx.amount for tx in transactions]
        recipients = {tx.recipient_id for tx in transactions if tx.recipient_id}
        senders = {tx.sender_id for tx in transactions if tx.sender_id}
        owners = {tx.owner_key for tx in transactions if tx.owner_key}
        payment_methods = Counter(tx.payment_method for tx in transactions if tx.payment_method)
        ibans = Counter(tx.recipient_iban for tx in transactions if tx.recipient_iban)
        return {
            "transaction_count": len(transactions),
            "distinct_recipients": len(recipients),
            "distinct_senders": len(senders),
            "distinct_owner_keys": len(owners),
            "mean_amount": mean(amounts),
            "p90_amount": quantile(amounts, 0.90),
            "payment_methods": payment_methods.most_common(6),
            "dominant_recipient_ibans": ibans.most_common(6),
        }

    def case_geo_summary(self, transactions: list[TransactionRecord]) -> dict[str, Any]:
        cities = []
        novel_count = 0
        residence_cities = set()
        hours = []
        for tx in transactions:
            geo = self.geo_behavior_summary(tx)
            if geo["transaction_city"]:
                cities.append(geo["transaction_city"])
            if geo["location_novelty"] > 0:
                novel_count += 1
            if geo["residence_city"]:
                residence_cities.add(geo["residence_city"])
            if tx.timestamp:
                hours.append(tx.timestamp.hour)
        return {
            "distinct_transaction_cities": sorted(set(cities)),
            "novel_location_count": novel_count,
            "residence_cities": sorted(residence_cities),
            "hour_histogram": Counter(hours).most_common(8),
        }

    def case_graph_summary(self, transaction_ids: list[str]) -> dict[str, Any]:
        total_neighbors = 0
        attribute_counts: Counter[str] = Counter()
        for tx_id in transaction_ids:
            if tx_id not in self.transactions_by_id:
                continue
            summary = self.graph.summary_for(tx_id)
            total_neighbors += summary["neighbor_count"]
            for attribute, count in summary["shared_attribute_counts"]:
                attribute_counts[attribute] += count
        return {
            "aggregate_neighbor_count": total_neighbors,
            "shared_attribute_counts": attribute_counts.most_common(8),
        }

    def _case_messages(
        self,
        transactions: list[TransactionRecord],
        *,
        lookback_hours: int,
        max_items: int,
    ) -> list[MessageEvent]:
        dedup: dict[str, MessageEvent] = {}
        for tx in transactions:
            for event in self.linked_messages_for(tx, lookback_hours=lookback_hours, max_items=max_items):
                dedup[event.message_id] = event
        ranked = sorted(dedup.values(), key=lambda item: (item.risk_score, _timestamp_sort_value(item.timestamp)), reverse=True)
        return ranked[:max_items]

    def _case_audio(
        self,
        transactions: list[TransactionRecord],
        *,
        lookback_hours: int,
        max_items: int,
    ) -> list[AudioAsset]:
        dedup: dict[str, AudioAsset] = {}
        for tx in transactions:
            for asset in self.linked_audio_for(tx, lookback_hours=lookback_hours, max_items=max_items):
                dedup[asset.asset_id] = asset
        ranked = sorted(dedup.values(), key=lambda item: _timestamp_sort_value(item.timestamp), reverse=True)
        return ranked[:max_items]


def build_feature_store(dataset: NormalizedDataset, config: AppConfig) -> FeatureStore:
    users_by_owner = {user.owner_key: user for user in dataset.users}
    users_by_iban = {user.iban: user for user in dataset.users if user.iban}

    owner_aliases: dict[str, set[str]] = defaultdict(set)
    sender_histories: dict[str, list[TransactionRecord]] = defaultdict(list)
    recipient_histories: dict[str, list[TransactionRecord]] = defaultdict(list)
    owner_histories: dict[str, list[TransactionRecord]] = defaultdict(list)
    pair_histories: dict[tuple[str, str], list[TransactionRecord]] = defaultdict(list)

    for tx in dataset.transactions:
        sender_owner = users_by_iban.get(tx.sender_iban)
        recipient_owner = users_by_iban.get(tx.recipient_iban)
        if sender_owner is not None:
            tx.owner_key = sender_owner.owner_key
            tx.direction = "outgoing"
            owner_aliases[sender_owner.owner_key].add(tx.sender_id)
        elif recipient_owner is not None:
            tx.owner_key = recipient_owner.owner_key
            tx.direction = "incoming"
            owner_aliases[recipient_owner.owner_key].add(tx.recipient_id)
        if sender_owner is not None and recipient_owner is not None and sender_owner.owner_key != recipient_owner.owner_key:
            tx.counterparty_owner_key = recipient_owner.owner_key
        elif recipient_owner is not None and tx.direction == "incoming":
            tx.counterparty_owner_key = sender_owner.owner_key if sender_owner else None

        sender_histories[tx.sender_id].append(tx)
        if tx.recipient_id:
            recipient_histories[tx.recipient_id].append(tx)
        if tx.owner_key:
            owner_histories[tx.owner_key].append(tx)
        pair_histories[(tx.sender_id, tx.recipient_id)].append(tx)

    signature_to_owner: dict[str, str] = {}
    for owner_key, aliases in owner_aliases.items():
        for alias in aliases:
            signature = entity_signature(alias)
            if signature and signature not in signature_to_owner:
                signature_to_owner[signature] = owner_key

    locations_by_owner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in dataset.locations:
        owner_key = signature_to_owner.get(entity_signature(event.biotag))
        event.owner_key = owner_key
        if owner_key:
            locations_by_owner[owner_key].append(
                {
                    "city": normalize_text(event.city),
                    "lat": event.lat,
                    "lng": event.lng,
                    "timestamp": event.timestamp,
                }
            )

    messages_by_owner: dict[str, list[MessageEvent]] = defaultdict(list)
    for event in dataset.messages:
        owner_key = _match_message_owner(event, dataset.users)
        event.owner_key = owner_key
        _score_message(event)
        if owner_key:
            messages_by_owner[owner_key].append(event)

    audio_by_owner: dict[str, list[AudioAsset]] = defaultdict(list)
    for asset in dataset.audio_assets:
        owner_key = _match_audio_owner(asset, dataset.users)
        asset.owner_key = owner_key
        if owner_key:
            audio_by_owner[owner_key].append(asset)

    for events in messages_by_owner.values():
        events.sort(key=lambda item: _timestamp_sort_value(item.timestamp))
    for events in audio_by_owner.values():
        events.sort(key=lambda item: _timestamp_sort_value(item.timestamp))
    for events in locations_by_owner.values():
        events.sort(key=lambda item: _timestamp_sort_value(item["timestamp"]))

    dataset_amounts = [tx.amount for tx in dataset.transactions]
    graph = GraphIndex(dataset.transactions)
    dataset_stats = {
        "transaction_count": len(dataset.transactions),
        "user_count": len(dataset.users),
        "message_count": len(dataset.messages),
        "audio_count": len(dataset.audio_assets),
        "location_count": len(dataset.locations),
        "high_value_amount": quantile(dataset_amounts, config.seeds.high_value_quantile),
        "message_density": len(dataset.messages) / max(1, len(dataset.transactions)),
        "audio_density": len(dataset.audio_assets) / max(1, len(dataset.transactions)),
    }

    return FeatureStore(
        dataset=dataset,
        users_by_owner=users_by_owner,
        users_by_iban=users_by_iban,
        transactions=sorted(dataset.transactions, key=lambda item: item.timestamp or datetime.min),
        transactions_by_id={tx.transaction_id: tx for tx in dataset.transactions},
        messages=dataset.messages,
        messages_by_owner=messages_by_owner,
        audio_assets=dataset.audio_assets,
        audio_by_owner=audio_by_owner,
        locations_by_owner=locations_by_owner,
        owner_aliases=owner_aliases,
        sender_histories=sender_histories,
        recipient_histories=recipient_histories,
        owner_histories=owner_histories,
        pair_histories=pair_histories,
        dataset_stats=dataset_stats,
        graph=graph,
    )


def _match_message_owner(event: MessageEvent, users: list[UserProfile]) -> str | None:
    haystack = " ".join([event.recipient_hint, event.subject, event.body_preview]).lower()
    haystack_tokens = name_tokens(haystack)
    best_key: str | None = None
    best_score = 0.0
    for user in users:
        score = 0.0
        user_tokens = name_tokens(user.full_name)
        if user_tokens and user_tokens.issubset(haystack_tokens):
            score += 2.2
        if normalize_text(user.first_name) in haystack_tokens:
            score += 1.4
        if normalize_text(user.last_name) in haystack_tokens:
            score += 1.4
        if user.owner_key and user.owner_key in normalize_text(haystack):
            score += 1.2
        if score > best_score:
            best_key = user.owner_key
            best_score = score
    return best_key if best_score >= 1.5 else None


def _match_audio_owner(asset: AudioAsset, users: list[UserProfile]) -> str | None:
    if not asset.owner_hint:
        return None
    for user in users:
        if asset.owner_hint == user.owner_key:
            return user.owner_key
    asset_tokens = set(asset.owner_hint.split("_"))
    best_key: str | None = None
    best_overlap = 0
    for user in users:
        overlap = len(asset_tokens & name_tokens(user.full_name))
        if overlap > best_overlap:
            best_overlap = overlap
            best_key = user.owner_key
    return best_key if best_overlap >= 2 else None


def _score_message(event: MessageEvent) -> None:
    text = f"{event.subject} {event.body_text}".lower()
    score = 0.0
    reasons: list[str] = []
    for label, terms in KEYWORD_GROUPS.items():
        if any(term in text for term in terms):
            score += 0.12
            reasons.append(label)
    url_domains = {extract_domain(url) for url in event.urls if extract_domain(url)}
    if event.sender_domain and url_domains and event.sender_domain not in url_domains:
        score += 0.18
        reasons.append("domain_mismatch")
    if event.channel == "sms" and event.urls:
        score += 0.08
        reasons.append("sms_link")
    event.risk_score = min(1.0, score)
    event.risk_reasons = reasons[:6]


def _within_hours(anchor: datetime | None, event_time: datetime | None, hours: int) -> bool:
    if anchor is None or event_time is None:
        return False
    delta = anchor - event_time
    return 0 <= delta.total_seconds() <= hours * 3600


def _prior_records(records: list[TransactionRecord], timestamp: datetime | None, tx_id: str) -> list[TransactionRecord]:
    if timestamp is None:
        return [item for item in records if item.transaction_id != tx_id]
    return [item for item in records if item.transaction_id != tx_id and item.timestamp and item.timestamp < timestamp]


def _count_within_hours(records: list[TransactionRecord], timestamp: datetime | None, hours: float) -> int:
    if timestamp is None:
        return 0
    return sum(
        1
        for item in records
        if item.timestamp and 0 <= (timestamp - item.timestamp).total_seconds() <= hours * 3600
    )


def _transaction_city(location_label: str) -> str:
    if not location_label:
        return ""
    return normalize_text(location_label.split(" - ", 1)[0])


def _timestamp_sort_value(value: datetime | None) -> float:
    return value.timestamp() if value is not None else float("-inf")
