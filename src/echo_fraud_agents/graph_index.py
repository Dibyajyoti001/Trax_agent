from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from echo_fraud_agents.models import TransactionRecord
from echo_fraud_agents.utils import normalize_text


@dataclass(slots=True)
class GraphNeighbor:
    transaction_id: str
    shared_entities: list[str]
    edge_score: float


class GraphIndex:
    def __init__(self, transactions: list[TransactionRecord]) -> None:
        self.transactions_by_id = {tx.transaction_id: tx for tx in transactions}
        self.entity_maps: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for tx in transactions:
            self._add("sender_id", tx.sender_id, tx.transaction_id)
            self._add("recipient_id", tx.recipient_id, tx.transaction_id)
            self._add("owner_key", tx.owner_key or "", tx.transaction_id)
            self._add("counterparty_owner", tx.counterparty_owner_key or "", tx.transaction_id)
            self._add("sender_iban", tx.sender_iban, tx.transaction_id)
            self._add("recipient_iban", tx.recipient_iban, tx.transaction_id)
            self._add("payment_method", tx.payment_method, tx.transaction_id)
            self._add("merchant_hint", tx.merchant_hint, tx.transaction_id)
            self._add("location_city", _location_city(tx.location_label), tx.transaction_id)

    def summary_for(self, transaction_id: str) -> dict[str, Any]:
        tx = self.transactions_by_id[transaction_id]
        counts: dict[str, int] = {}
        total_neighbors = 0
        for field_name, value in _graph_values(tx):
            if not value:
                continue
            neighbors = self.entity_maps[field_name][value]
            count = max(0, len(neighbors) - 1)
            if count:
                counts[field_name] = count
                total_neighbors += count
        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return {
            "neighbor_count": total_neighbors,
            "shared_attribute_counts": ranked[:5],
        }

    def expand(
        self,
        anchor: TransactionRecord,
        *,
        max_size: int,
        time_window_hours: int,
    ) -> list[GraphNeighbor]:
        candidates: dict[str, list[str]] = defaultdict(list)
        for field_name, value in _graph_values(anchor):
            if not value:
                continue
            for neighbor_id in self.entity_maps[field_name][value]:
                if neighbor_id == anchor.transaction_id:
                    continue
                other = self.transactions_by_id[neighbor_id]
                if not _within_hours(anchor.timestamp, other.timestamp, time_window_hours):
                    continue
                candidates[neighbor_id].append(f"{field_name}:{value}")

        ranked: list[GraphNeighbor] = []
        for neighbor_id, shared in candidates.items():
            score = float(len(shared))
            if any(label.startswith("owner_key:") for label in shared):
                score += 0.75
            if any(label.startswith("recipient_iban:") for label in shared):
                score += 0.5
            ranked.append(GraphNeighbor(transaction_id=neighbor_id, shared_entities=shared[:8], edge_score=score))
        ranked.sort(key=lambda item: item.edge_score, reverse=True)
        return ranked[:max_size]

    def _add(self, field_name: str, value: str, transaction_id: str) -> None:
        normalized = normalize_text(value)
        if normalized:
            self.entity_maps[field_name][normalized].add(transaction_id)


def _location_city(location_label: str) -> str:
    if not location_label:
        return ""
    head = location_label.split(" - ", 1)[0]
    return normalize_text(head)


def _graph_values(tx: TransactionRecord) -> list[tuple[str, str]]:
    return [
        ("sender_id", tx.sender_id),
        ("recipient_id", tx.recipient_id),
        ("owner_key", tx.owner_key or ""),
        ("counterparty_owner", tx.counterparty_owner_key or ""),
        ("sender_iban", tx.sender_iban),
        ("recipient_iban", tx.recipient_iban),
        ("payment_method", tx.payment_method),
        ("merchant_hint", tx.merchant_hint),
        ("location_city", _location_city(tx.location_label)),
    ]


def _within_hours(left: datetime | None, right: datetime | None, hours: int) -> bool:
    if left is None or right is None:
        return True
    return abs((left - right).total_seconds()) <= hours * 3600
