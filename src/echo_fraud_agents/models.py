from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class DatasetBundle:
    name: str
    slug: str
    source_label: str
    files: dict[str, bytes]


@dataclass(slots=True)
class DatasetManifest:
    name: str
    slug: str
    source_label: str
    modalities: list[str]
    record_counts: dict[str, int]
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TransactionRecord:
    transaction_id: str
    sender_id: str
    recipient_id: str
    transaction_type: str
    amount: float
    location_label: str
    payment_method: str
    sender_iban: str
    recipient_iban: str
    balance_after: float
    description: str
    timestamp: datetime | None
    owner_key: str | None = None
    counterparty_owner_key: str | None = None
    direction: str = "external"
    merchant_hint: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UserProfile:
    owner_key: str
    first_name: str
    last_name: str
    full_name: str
    birth_year: int | None
    salary: float
    job: str
    iban: str
    residence_city: str
    residence_lat: float
    residence_lng: float
    description: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MessageEvent:
    message_id: str
    channel: str
    sender_label: str
    sender_domain: str
    recipient_hint: str
    owner_key: str | None
    timestamp: datetime | None
    subject: str
    body_text: str
    body_preview: str
    urls: list[str]
    risk_score: float = 0.0
    risk_reasons: list[str] = field(default_factory=list)
    raw_text: str = ""


@dataclass(slots=True)
class AudioAsset:
    asset_id: str
    filename: str
    relative_path: str
    owner_key: str | None
    owner_hint: str
    timestamp: datetime | None
    extension: str
    size_bytes: int


@dataclass(slots=True)
class LocationEvent:
    event_id: str
    biotag: str
    owner_key: str | None
    city: str
    lat: float
    lng: float
    timestamp: datetime | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizedDataset:
    bundle: DatasetBundle
    manifest: DatasetManifest
    schema: dict[str, Any]
    transactions: list[TransactionRecord]
    users: list[UserProfile]
    messages: list[MessageEvent]
    audio_assets: list[AudioAsset]
    locations: list[LocationEvent]


@dataclass(slots=True)
class CandidateSeed:
    transaction_id: str
    score: float
    reasons: list[str]
    signal_breakdown: dict[str, float]
    economic_severity: float
    selected: bool = False


@dataclass(slots=True)
class CaseCandidate:
    case_id: str
    case_type: str
    anchor_transaction_ids: list[str]
    member_transaction_ids: list[str]
    seed_transaction_ids: list[str]
    owner_keys: list[str]
    shared_entities: list[str]
    modality_coverage: list[str]
    priority_hint: float
    economic_severity: float
    earliest_timestamp: datetime | None
    latest_timestamp: datetime | None
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CaseDiscoveryDecision:
    case_type: str
    priority: float
    mandatory_tribunal: bool
    required_routes: list[str]
    suspected_mechanism: str
    reason: str
    model: str | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RoutingDecision:
    route: str
    routes: list[str]
    priority: float
    reason: str
    mandatory_tribunal: bool = False
    escalate_on_borderline: bool = True
    case_type: str = ""
    model: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SpecialistOpinion:
    stage: str
    score: float
    confidence: float
    label: str
    rationale: str
    suspicious_entities: list[str]
    expand_neighbors: bool
    model: str | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ClusterDecision:
    cluster_score: float
    confidence: float
    include_ids: list[str]
    suspicious_entities: list[str]
    rationale: str
    fraud_mechanism: str
    model: str | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JudgeDecision:
    fraud_mechanism: str
    final_score: float
    verdict: str
    include_ids: list[str]
    borderline_ids: list[str]
    exclude_ids: list[str]
    key_entities: list[str]
    rationale: str
    confidence: float
    model: str | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluatedTransaction:
    transaction: TransactionRecord
    candidate_seed: CandidateSeed | None
    case_ids: list[str] = field(default_factory=list)
    case_types: list[str] = field(default_factory=list)
    routing: RoutingDecision | None = None
    specialists: list[SpecialistOpinion] = field(default_factory=list)
    cluster: ClusterDecision | None = None
    judge: JudgeDecision | None = None
    selected: bool = False
    selected_via_cluster: bool = False
    fraud_mechanisms: list[str] = field(default_factory=list)
    rationales: list[str] = field(default_factory=list)
    tribunal_scores: list[float] = field(default_factory=list)


@dataclass(slots=True)
class CaseReview:
    case: CaseCandidate
    discovery: CaseDiscoveryDecision | None
    routing: RoutingDecision | None
    specialists: list[SpecialistOpinion]
    cluster: ClusterDecision | None
    tribunal: JudgeDecision | None
    second_pass: JudgeDecision | None
    selected_ids: list[str] = field(default_factory=list)
    borderline_ids: list[str] = field(default_factory=list)
    excluded_ids: list[str] = field(default_factory=list)
