from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from echo_fraud_agents.models import CaseCandidate, ClusterDecision, JudgeDecision, SpecialistOpinion
from echo_fraud_agents.utils import extract_domain


@dataclass(slots=True)
class FraudMotif:
    category: str
    fraud_mechanism: str
    rationale: str
    suspicious_entities: list[str]
    confidence: float
    case_id: str
    included_ids: list[str]


class FraudMemory:
    def __init__(self, max_entries: int = 180) -> None:
        self.max_entries = max_entries
        self.motifs: deque[FraudMotif] = deque(maxlen=max_entries)
        self.category_counter: Counter[str] = Counter()
        self.entity_counter: Counter[str] = Counter()
        self.domain_counter: Counter[str] = Counter()
        self.case_type_counter: Counter[str] = Counter()

    def remember_case(
        self,
        *,
        case: CaseCandidate,
        tribunal: JudgeDecision,
        specialists: list[SpecialistOpinion],
        cluster: ClusterDecision | None,
    ) -> None:
        if tribunal.verdict != "include" or tribunal.final_score < 0.5:
            return
        categories = [opinion.label for opinion in specialists if opinion.label and opinion.label not in {"normal", "uncertain"}]
        category = categories[0] if categories else case.case_type
        suspicious_entities = list(dict.fromkeys(tribunal.key_entities or []))
        for opinion in specialists:
            suspicious_entities.extend(opinion.suspicious_entities)
        if cluster is not None:
            suspicious_entities.extend(cluster.suspicious_entities)
        suspicious_entities.extend(case.shared_entities)
        suspicious_entities = list(dict.fromkeys(entity for entity in suspicious_entities if entity))[:16]

        fraud_mechanism = tribunal.fraud_mechanism
        if not fraud_mechanism and cluster is not None:
            fraud_mechanism = cluster.fraud_mechanism
        motif = FraudMotif(
            category=category,
            fraud_mechanism=fraud_mechanism,
            rationale=tribunal.rationale,
            suspicious_entities=suspicious_entities,
            confidence=max(tribunal.final_score, tribunal.confidence),
            case_id=case.case_id,
            included_ids=tribunal.include_ids[:18],
        )
        self.motifs.append(motif)
        self.category_counter[category] += 1
        self.case_type_counter[case.case_type] += 1
        for entity in suspicious_entities:
            self.entity_counter[entity] += 1
            domain = extract_domain(entity)
            if "." in entity or domain:
                self.domain_counter[domain or entity] += 1

    def summary(self) -> dict[str, object]:
        recent = list(self.motifs)[-10:]
        return {
            "recent_categories": [item.category for item in recent],
            "recent_case_ids": [item.case_id for item in recent],
            "recent_mechanisms": [item.fraud_mechanism[:120] for item in recent],
            "hot_entities": [item for item, _ in self.entity_counter.most_common(12)],
            "hot_domains": [item for item, _ in self.domain_counter.most_common(8)],
            "category_counts": dict(self.category_counter.most_common(8)),
            "case_type_counts": dict(self.case_type_counter.most_common(8)),
        }
