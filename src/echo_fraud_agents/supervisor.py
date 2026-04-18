from __future__ import annotations

from echo_fraud_agents.config import AppConfig
from echo_fraud_agents.llm_client import OpenRouterLLMClient, coerce_bool, coerce_float, coerce_list
from echo_fraud_agents.models import CaseDiscoveryDecision, RoutingDecision


ALLOWED_CASE_TYPES = {
    "recipient_campaign",
    "victim_social_engineering",
    "owner_behavior_shift",
    "geo_behavior_case",
    "graph_cluster",
    "high_value_anomaly",
    "uncertain_case",
}

ALLOWED_ROUTES = {
    "ignore",
    "communication",
    "transaction_pattern",
    "geo_profile",
    "cluster",
    "audio",
    "judge",
}


def _normalize_case_type(value: object) -> str:
    text = str(value or "uncertain_case").strip().lower()
    aliases = {
        "recipient_siphon": "recipient_campaign",
        "recipient_cluster": "recipient_campaign",
        "social_engineering": "victim_social_engineering",
        "phishing_case": "victim_social_engineering",
        "owner_shift": "owner_behavior_shift",
        "behavior_shift": "owner_behavior_shift",
        "geo_behavior_shift": "geo_behavior_case",
        "cluster_case": "graph_cluster",
        "high_value_case": "high_value_anomaly",
        "uncertain": "uncertain_case",
    }
    normalized = aliases.get(text, text)
    return normalized if normalized in ALLOWED_CASE_TYPES else "uncertain_case"


def _normalize_route(value: object) -> str:
    text = str(value or "judge").strip().lower()
    mapping = {
        "comm": "communication",
        "transaction": "transaction_pattern",
        "tx_pattern": "transaction_pattern",
        "geo": "geo_profile",
        "profile": "geo_profile",
        "tribunal": "judge",
        "final_judge": "judge",
    }
    normalized = mapping.get(text, text)
    return normalized if normalized in ALLOWED_ROUTES else "judge"


def _normalize_routes(values: list[str]) -> list[str]:
    routes = []
    for value in values:
        normalized = _normalize_route(value)
        if normalized not in routes:
            routes.append(normalized)
    return routes


class CaseDiscoveryAgent:
    def __init__(self, config: AppConfig, llm_client: OpenRouterLLMClient) -> None:
        self.config = config
        self.llm_client = llm_client

    def discover(self, *, session_id: str, payload: dict) -> CaseDiscoveryDecision | None:
        data = self.llm_client.call_role_json(
            session_id=session_id,
            stage="case_discovery",
            models=self.config.models.supervisor,
            system_prompt=(
                "You are a fraud case discovery analyst in a bounded multimodal investigation system. "
                "Interpret the candidate case as a possible fraud mechanism or campaign, not as isolated rows. "
                "Return JSON with keys case_type, priority, mandatory_tribunal, required_routes, suspected_mechanism, reason. "
                "case_type must be one of: recipient_campaign, victim_social_engineering, owner_behavior_shift, "
                "geo_behavior_case, graph_cluster, high_value_anomaly, uncertain_case. "
                "required_routes should be a list using only: communication, transaction_pattern, geo_profile, cluster, audio, judge."
            ),
            payload=payload,
            max_tokens=self.config.token_limits.supervisor,
        )
        if data.get("_llm_failed"):
            return None
        return CaseDiscoveryDecision(
            case_type=_normalize_case_type(data.get("case_type")),
            priority=coerce_float(data.get("priority"), default=0.0),
            mandatory_tribunal=coerce_bool(data.get("mandatory_tribunal"), default=False),
            required_routes=_normalize_routes(coerce_list(data.get("required_routes"))),
            suspected_mechanism=str(data.get("suspected_mechanism", "")),
            reason=str(data.get("reason", "")),
            model=data.get("_llm_model"),
            raw=data,
        )


class TriageAgent:
    def __init__(self, config: AppConfig, llm_client: OpenRouterLLMClient) -> None:
        self.config = config
        self.llm_client = llm_client

    def route(self, *, session_id: str, payload: dict) -> RoutingDecision | None:
        data = self.llm_client.call_role_json(
            session_id=session_id,
            stage="case_triage",
            models=self.config.models.supervisor,
            system_prompt=(
                "You are the fraud triage coordinator for a bounded LLM-led investigation network. "
                "Choose a primary route and any additional specialist routes. "
                "Strong campaign or high-value cases should be marked mandatory_tribunal. "
                "Return JSON with keys route, additional_routes, priority, mandatory_tribunal, escalate_on_borderline, reason. "
                "route and additional_routes must use only: ignore, communication, transaction_pattern, geo_profile, cluster, audio, judge."
            ),
            payload=payload,
            max_tokens=self.config.token_limits.supervisor,
        )
        if data.get("_llm_failed"):
            return None
        route = _normalize_route(data.get("route", "judge"))
        routes = _normalize_routes([route, *coerce_list(data.get("additional_routes"))])
        if route == "ignore":
            actionable = [item for item in routes if item != "ignore"]
            if actionable:
                route = actionable[0]
        return RoutingDecision(
            route=route,
            routes=routes,
            priority=coerce_float(data.get("priority"), default=0.0),
            reason=str(data.get("reason", "")),
            mandatory_tribunal=coerce_bool(data.get("mandatory_tribunal"), default=False),
            escalate_on_borderline=coerce_bool(data.get("escalate_on_borderline"), default=True),
            case_type=str(payload.get("case_type", "")),
            model=data.get("_llm_model"),
            raw=data,
        )
