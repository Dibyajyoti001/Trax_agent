from __future__ import annotations

from echo_fraud_agents.config import AppConfig
from echo_fraud_agents.feature_store import FeatureStore
from echo_fraud_agents.models import CandidateSeed, TransactionRecord


def generate_high_recall_seeds(store: FeatureStore, config: AppConfig) -> dict[str, CandidateSeed]:
    seeds: dict[str, CandidateSeed] = {}
    high_value_amount = max(1.0, store.dataset_stats["high_value_amount"])
    for tx in store.transactions:
        signal_breakdown: dict[str, float] = {}
        reasons: list[str] = []

        pattern = store.transaction_pattern_summary(tx)
        geo = store.geo_behavior_summary(tx)
        graph = store.graph_summary(tx)
        linked_messages = store.linked_messages_for(
            tx,
            lookback_hours=config.linking.message_lookback_hours,
            max_items=config.budgets.max_messages_per_case,
        )

        amount_score = 0.0
        if tx.amount >= high_value_amount:
            amount_score += 0.16
            reasons.append("high_value")
        if pattern["p90_amount"] and tx.amount >= pattern["p90_amount"] * config.seeds.high_amount_multiplier:
            amount_score += 0.18
            reasons.append("amount_vs_sender_history")
        if tx.amount >= config.seeds.new_recipient_amount and pattern["pair_prior_count"] == 0:
            amount_score += 0.16
            reasons.append("new_recipient")

        recipient_score = 0.0
        recipient_history = store.recipient_histories.get(tx.recipient_id, [])
        distinct_senders = len({item.sender_id for item in recipient_history})
        if distinct_senders >= config.seeds.recipient_fan_in_threshold:
            recipient_score += 0.16
            reasons.append("recipient_fan_in")

        message_score = 0.0
        if linked_messages:
            top_message_risk = max(item.risk_score for item in linked_messages)
            if top_message_risk >= config.seeds.message_risk_floor:
                message_score += min(0.28, top_message_risk)
                reasons.append("linked_message_risk")

        geo_score = 0.0
        if geo["location_novelty"] > 0:
            geo_score += 0.12
            reasons.append("location_novelty")
        if geo["current_hour_seen"] == 0 and tx.timestamp is not None:
            geo_score += 0.08
            reasons.append("hour_novelty")

        graph_score = 0.0
        if graph["neighbor_count"] >= config.seeds.graph_neighbor_floor:
            graph_score += 0.10
            reasons.append("graph_neighbors")

        drift_score = 0.0
        if pattern["current_payment_method_prior"] == 0 and tx.payment_method:
            drift_score += 0.08
            reasons.append("payment_shift")
        if pattern["recent_30m_count"] >= 2 or pattern["recent_3h_count"] >= 4:
            drift_score += 0.08
            reasons.append("velocity")

        signal_breakdown.update(
            {
                "amount": amount_score,
                "recipient": recipient_score,
                "message": message_score,
                "geo": geo_score,
                "graph": graph_score,
                "drift": drift_score,
            }
        )
        final_score = min(1.0, sum(signal_breakdown.values()))
        seeds[tx.transaction_id] = CandidateSeed(
            transaction_id=tx.transaction_id,
            score=final_score,
            reasons=reasons[:8],
            signal_breakdown=signal_breakdown,
            economic_severity=store.economic_severity(tx),
            selected=False,
        )

    _mark_selected(seeds, store.transactions, config)
    return seeds


def _mark_selected(seeds: dict[str, CandidateSeed], transactions: list[TransactionRecord], config: AppConfig) -> None:
    ranked = sorted(seeds.values(), key=lambda item: (item.score, item.economic_severity), reverse=True)
    target_count = int(len(transactions) * config.budgets.seed_fraction)
    target_count = max(config.budgets.min_seed_count, target_count)
    target_count = min(config.budgets.max_seed_count, max(target_count, config.budgets.min_seed_count))
    for item in ranked[:target_count]:
        item.selected = True
    for item in ranked:
        if item.signal_breakdown.get("message", 0.0) >= 0.22 or item.signal_breakdown.get("amount", 0.0) >= 0.28:
            item.selected = True
