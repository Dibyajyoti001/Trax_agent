from __future__ import annotations

from echo_fraud_agents.config import AppConfig
from echo_fraud_agents.llm_client import OpenRouterLLMClient, coerce_bool, coerce_float, coerce_list
from echo_fraud_agents.models import SpecialistOpinion


def _normalize_stage_result(stage: str, data: dict, *, label_key: str) -> SpecialistOpinion | None:
    if data.get("_llm_failed"):
        return None
    return SpecialistOpinion(
        stage=stage,
        score=coerce_float(data.get("score"), default=0.0),
        confidence=coerce_float(data.get("confidence"), default=0.0),
        label=str(data.get(label_key, data.get("fraud_type", "uncertain")) or "uncertain").strip().lower(),
        rationale=str(data.get("rationale", "")),
        suspicious_entities=coerce_list(data.get("suspicious_entities") or data.get("red_flags") or data.get("key_entities")),
        expand_neighbors=coerce_bool(data.get("expand_neighbors"), default=False),
        model=data.get("_llm_model"),
        raw=data,
    )


class CommunicationSpecialist:
    def __init__(self, config: AppConfig, llm_client: OpenRouterLLMClient) -> None:
        self.config = config
        self.llm_client = llm_client

    def review(self, *, session_id: str, payload: dict) -> SpecialistOpinion | None:
        data = self.llm_client.call_role_json(
            session_id=session_id,
            stage="communication_specialist",
            models=self.config.models.communication,
            system_prompt=(
                "You are the communication specialist in a multimodal fraud tribunal. "
                "Reason over the full case file, linked messages, message timing, sender domains, and payment pressure. "
                "Classify likely mechanisms such as phishing, verification_scam, payment_redirection, urgency_coercion, credential_theft, normal, or uncertain. "
                "Return JSON with keys score, confidence, fraud_type, rationale, red_flags, suspicious_entities, expand_neighbors."
            ),
            payload=payload,
            max_tokens=self.config.token_limits.specialist,
        )
        return _normalize_stage_result("communication", data, label_key="fraud_type")


class TransactionPatternSpecialist:
    def __init__(self, config: AppConfig, llm_client: OpenRouterLLMClient) -> None:
        self.config = config
        self.llm_client = llm_client

    def review(self, *, session_id: str, payload: dict) -> SpecialistOpinion | None:
        data = self.llm_client.call_role_json(
            session_id=session_id,
            stage="transaction_pattern_specialist",
            models=self.config.models.transaction_pattern,
            system_prompt=(
                "You are the transaction-pattern specialist in a fraud investigation network. "
                "Reason over the case file, recipient concentration, sender cohorts, amount regularity, balance drains, "
                "velocity spikes, payroll-like diversion, and campaign-style payment flows. "
                "Return JSON with keys score, confidence, pattern_type, rationale, suspicious_entities, expand_neighbors."
            ),
            payload=payload,
            max_tokens=self.config.token_limits.specialist,
        )
        return _normalize_stage_result("transaction_pattern", data, label_key="pattern_type")


class GeoBehaviorSpecialist:
    def __init__(self, config: AppConfig, llm_client: OpenRouterLLMClient) -> None:
        self.config = config
        self.llm_client = llm_client

    def review(self, *, session_id: str, payload: dict) -> SpecialistOpinion | None:
        data = self.llm_client.call_role_json(
            session_id=session_id,
            stage="geo_behavior_specialist",
            models=self.config.models.geo_behavior,
            system_prompt=(
                "You are the geo-behavior specialist in a fraud tribunal. "
                "Reason over case-level movement patterns, location novelty, residence mismatch, "
                "time-of-day behavior, and travel inconsistency across linked transactions. "
                "Return JSON with keys score, confidence, pattern_type, rationale, suspicious_entities, expand_neighbors."
            ),
            payload=payload,
            max_tokens=self.config.token_limits.specialist,
        )
        return _normalize_stage_result("geo_behavior", data, label_key="pattern_type")


class AudioSpecialist:
    def __init__(self, config: AppConfig, llm_client: OpenRouterLLMClient) -> None:
        self.config = config
        self.llm_client = llm_client

    def review(self, *, session_id: str, payload: dict) -> SpecialistOpinion | None:
        data = self.llm_client.call_role_json(
            session_id=session_id,
            stage="audio_specialist",
            models=self.config.models.audio,
            system_prompt=(
                "You are the audio specialist in a fraud investigation system. "
                "Reason from audio metadata, ownership linkage, timing, and surrounding case context. "
                "If no transcript is present, infer only from metadata and links. "
                "Return JSON with keys score, confidence, pattern_type, rationale, suspicious_entities, expand_neighbors."
            ),
            payload=payload,
            max_tokens=self.config.token_limits.specialist,
        )
        return _normalize_stage_result("audio", data, label_key="pattern_type")
