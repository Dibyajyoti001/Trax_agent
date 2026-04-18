from __future__ import annotations

from echo_fraud_agents.config import AppConfig
from echo_fraud_agents.llm_client import OpenRouterLLMClient, coerce_float, coerce_list
from echo_fraud_agents.models import JudgeDecision


def _normalize_verdict(value: object) -> str:
    text = str(value or "exclude").strip().lower()
    mapping = {
        "keep": "include",
        "accept": "include",
        "fraud": "include",
        "reject": "exclude",
        "drop": "exclude",
        "maybe": "borderline",
        "review": "borderline",
    }
    normalized = mapping.get(text, text)
    return normalized if normalized in {"include", "exclude", "borderline"} else "exclude"


class JudgeAgent:
    def __init__(self, config: AppConfig, llm_client: OpenRouterLLMClient) -> None:
        self.config = config
        self.llm_client = llm_client

    def review(self, *, session_id: str, payload: dict) -> JudgeDecision | None:
        return self._review_with_stage(
            session_id=session_id,
            payload=payload,
            stage="fraud_tribunal",
            system_prompt=(
                "You are the final Fraud Tribunal in a bounded multimodal fraud investigation system. "
                "Reason over cases and campaigns, not isolated rows. "
                "Prioritize recall, but keep your reasoning evidence-grounded and coherent. "
                "If the same fraud mechanism plausibly explains multiple linked transactions, include the linked set. "
                "Use borderline when suspicion is real but support is incomplete. "
                "Return valid JSON only with keys fraud_mechanism, final_score, verdict, include_ids, borderline_ids, "
                "exclude_ids, key_entities, rationale, confidence."
            ),
        )

    def review_second_pass(self, *, session_id: str, payload: dict) -> JudgeDecision | None:
        return self._review_with_stage(
            session_id=session_id,
            payload=payload,
            stage="second_pass_tribunal",
            system_prompt=(
                "You are the second-pass Fraud Tribunal. "
                "Resolve high-priority borderline or conflicted fraud cases. "
                "Reason over the original case file, first tribunal output, specialist conflicts, and campaign structure. "
                "Prefer inclusion when the case shows repeated entities, linked victims, cross-modal evidence, or economic severity. "
                "Return valid JSON only with keys fraud_mechanism, final_score, verdict, include_ids, borderline_ids, "
                "exclude_ids, key_entities, rationale, confidence."
            ),
        )

    def _review_with_stage(
        self,
        *,
        session_id: str,
        payload: dict,
        stage: str,
        system_prompt: str,
    ) -> JudgeDecision | None:
        data = self.llm_client.call_role_json(
            session_id=session_id,
            stage=stage,
            models=self.config.models.judge,
            system_prompt=system_prompt,
            payload=payload,
            max_tokens=self.config.token_limits.judge,
        )
        if data.get("_llm_failed"):
            return None
        return JudgeDecision(
            fraud_mechanism=str(data.get("fraud_mechanism", "")),
            final_score=coerce_float(data.get("final_score"), default=0.0),
            verdict=_normalize_verdict(data.get("verdict")),
            include_ids=coerce_list(data.get("include_ids")),
            borderline_ids=coerce_list(data.get("borderline_ids")),
            exclude_ids=coerce_list(data.get("exclude_ids")),
            key_entities=coerce_list(data.get("key_entities") or data.get("suspicious_entities")),
            rationale=str(data.get("rationale", "")),
            confidence=coerce_float(data.get("confidence"), default=0.0),
            model=data.get("_llm_model"),
            raw=data,
        )
