from __future__ import annotations

from echo_fraud_agents.config import AppConfig
from echo_fraud_agents.feature_store import FeatureStore
from echo_fraud_agents.graph_index import GraphNeighbor
from echo_fraud_agents.llm_client import OpenRouterLLMClient, coerce_float, coerce_list
from echo_fraud_agents.models import ClusterDecision, TransactionRecord


class ClusterExpander:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def expand(self, store: FeatureStore, anchors: list[TransactionRecord]) -> list[dict]:
        all_neighbors = []
        seen_ids = set()
        for anchor in anchors:
            neighbors = store.graph.expand(
                anchor,
                max_size=self.config.budgets.max_cluster_size,
                time_window_hours=self.config.budgets.max_cluster_time_window_hours,
            )
            for neighbor in neighbors:
                if neighbor.transaction_id not in seen_ids:
                    seen_ids.add(neighbor.transaction_id)
                    all_neighbors.append(_neighbor_summary(store, neighbor))
            if len(all_neighbors) >= self.config.budgets.max_cluster_size * 2:
                break
        return all_neighbors[:self.config.budgets.max_cluster_size * 2]


class ClusterSpecialistAgent:
    def __init__(self, config: AppConfig, llm_client: OpenRouterLLMClient) -> None:
        self.config = config
        self.llm_client = llm_client

    def review(self, *, session_id: str, payload: dict) -> ClusterDecision | None:
        data = self.llm_client.call_role_json(
            session_id=session_id,
            stage="cluster_specialist",
            models=self.config.models.cluster,
            system_prompt=(
                "You are the cluster and campaign specialist in a fraud investigation system. "
                "Reason over linked transactions, shared entities, recipient concentration, owner overlap, and timing. "
                "Return JSON with keys cluster_score, confidence, include_ids, suspicious_entities, rationale, fraud_mechanism."
            ),
            payload=payload,
            max_tokens=self.config.token_limits.cluster,
        )
        if data.get("_llm_failed"):
            return None
        return ClusterDecision(
            cluster_score=coerce_float(data.get("cluster_score"), default=0.0),
            confidence=coerce_float(data.get("confidence"), default=0.0),
            include_ids=coerce_list(data.get("include_ids")),
            suspicious_entities=coerce_list(data.get("suspicious_entities") or data.get("key_entities")),
            rationale=str(data.get("rationale", "")),
            fraud_mechanism=str(data.get("fraud_mechanism", "")),
            model=data.get("_llm_model"),
            raw=data,
        )


def _neighbor_summary(store: FeatureStore, neighbor: GraphNeighbor) -> dict:
    tx = store.transactions_by_id[neighbor.transaction_id]
    return {
        "transaction_id": tx.transaction_id,
        "sender_id": tx.sender_id,
        "recipient_id": tx.recipient_id,
        "owner_key": tx.owner_key,
        "amount": tx.amount,
        "payment_method": tx.payment_method,
        "location": tx.location_label,
        "timestamp": tx.timestamp.isoformat() if tx.timestamp else None,
        "shared_entities": neighbor.shared_entities,
        "edge_score": neighbor.edge_score,
    }
