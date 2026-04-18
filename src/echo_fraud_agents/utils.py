from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse


URL_RE = re.compile(r"https?://[^\s<>\"]+")
EMAIL_RE = re.compile(r"([A-Z0-9._%+\-]+)@([A-Z0-9.\-]+\.[A-Z]{2,})", re.IGNORECASE)


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", value)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def slugify(value: str) -> str:
    return normalize_text(value).replace("_", "-") or "dataset"


def chunked(items: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), max(1, size)):
        yield items[start : start + size]


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    for parser in (datetime.fromisoformat,):
        try:
            dt = parser(text)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y%m%d_%H%M%S"):
        try:
            return datetime.strptime(text, pattern).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        dt = parsedate_to_datetime(text)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError, IndexError):
        return None


def safe_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(str(value).strip())
    except ValueError:
        return default


def quantile(values: Iterable[float], q: float) -> float:
    sample = sorted(float(v) for v in values if v is not None)
    if not sample:
        return 0.0
    if len(sample) == 1:
        return sample[0]
    q = min(1.0, max(0.0, q))
    index = (len(sample) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sample[lower]
    weight = index - lower
    return sample[lower] * (1 - weight) + sample[upper] * weight


def mean(values: Iterable[float]) -> float:
    sample = [float(v) for v in values if v is not None]
    return sum(sample) / len(sample) if sample else 0.0


def pstdev(values: Iterable[float]) -> float:
    sample = [float(v) for v in values if v is not None]
    if len(sample) < 2:
        return 0.0
    avg = mean(sample)
    return math.sqrt(sum((v - avg) ** 2 for v in sample) / len(sample))


def extract_urls(text: str) -> list[str]:
    return URL_RE.findall(text or "")


def extract_domain(value: str | None) -> str:
    if not value:
        return ""
    match = EMAIL_RE.search(value)
    if match:
        return normalize_text(match.group(2).lower().strip("."))
    parsed = urlparse(value if "://" in value else f"https://{value}")
    return normalize_text(parsed.netloc or parsed.path.split("/")[0])


def strip_html(value: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return re.sub(r"<[^>]+>", " ", value)
    soup = BeautifulSoup(value, "html.parser")
    return soup.get_text(" ", strip=True)


def compact_text(value: str, limit: int = 320) -> str:
    clean = re.sub(r"\s+", " ", value or "").strip()
    return clean[:limit]


def name_tokens(value: str) -> set[str]:
    return {token for token in normalize_text(value).split("_") if token}


def entity_signature(value: str | None) -> str:
    if not value:
        return ""
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", value.upper()) if part]
    if not parts:
        return ""
    return "-".join(parts[:2])


def file_basename(path: str) -> str:
    return PurePosixPath(path).name
