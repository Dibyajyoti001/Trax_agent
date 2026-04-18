from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from echo_fraud_agents.config import AppConfig
from echo_fraud_agents.feature_store import FeatureStore
from echo_fraud_agents.models import CandidateSeed, CaseCandidate


def build_case_candidates(
    store: FeatureStore,
    seeds: dict[str, CandidateSeed],
    config: AppConfig,
) -> list[CaseCandidate]:
    selected_seed_ids = {seed.transaction_id for seed in seeds.values() if seed.selected}
    if not selected_seed_ids:
        return []

    cases: list[CaseCandidate] = []
    seen_keys: set[tuple[str, tuple[str, ...]]] = set()

    recipient_groups: dict[str, list[str]] = defaultdict(list)
    owner_groups: dict[str, list[str]] = defaultdict(list)
    social_groups: dict[str, list[str]] = defaultdict(list)
    geo_groups: dict[str, list[str]] = defaultdict(list)

    for tx_id in selected_seed_ids:
        tx = store.transactions_by_id.get(tx_id)
        if tx is None:
            continue
        if tx.recipient_id:
            recipient_groups[f"recipient_id::{tx.recipient_id}"].append(tx_id)
        if tx.recipient_iban:
            recipient_groups[f"recipient_iban::{tx.recipient_iban}"].append(tx_id)
        if tx.owner_key:
            owner_groups[tx.owner_key].append(tx_id)
            linked_messages = store.linked_messages_for(
                tx,
                lookback_hours=config.linking.message_lookback_hours,
                max_items=config.budgets.max_messages_per_case,
            )
            if any(item.risk_score >= config.seeds.message_risk_floor for item in linked_messages):
                social_groups[tx.owner_key].append(tx_id)
            geo_summary = store.geo_behavior_summary(tx)
            if geo_summary["location_novelty"] > 0 or geo_summary["current_hour_seen"] == 0:
                geo_groups[tx.owner_key].append(tx_id)

    for recipient_key, tx_ids in recipient_groups.items():
        case = _recipient_campaign_case(store, seeds, config, recipient_key, tx_ids)
        _append_case(cases, seen_keys, case)

    for owner_key, tx_ids in owner_groups.items():
        case = _owner_behavior_case(store, seeds, config, owner_key, tx_ids)
        _append_case(cases, seen_keys, case)

    for owner_key, tx_ids in social_groups.items():
        case = _social_engineering_case(store, seeds, config, owner_key, tx_ids)
        _append_case(cases, seen_keys, case)

    for owner_key, tx_ids in geo_groups.items():
        case = _geo_case(store, seeds, config, owner_key, tx_ids)
        _append_case(cases, seen_keys, case)

    ranked_seeds = sorted(
        (seeds[tx_id] for tx_id in selected_seed_ids if tx_id in seeds),
        key=lambda item: (item.score, item.economic_severity),
        reverse=True,
    )
    for seed in ranked_seeds[: max(12, config.budgets.max_case_candidates // 3)]:
        case = _graph_case(store, seeds, config, seed.transaction_id)
        _append_case(cases, seen_keys, case)

    covered_transactions = {
        tx_id
        for case in cases
        for tx_id in case.member_transaction_ids
    }
    for seed in ranked_seeds:
        if seed.transaction_id in covered_transactions:
            continue
        case = _high_value_case(store, seeds, seed.transaction_id)
        _append_case(cases, seen_keys, case)

    cases.sort(key=lambda item: (item.priority_hint, item.economic_severity, len(item.member_transaction_ids)), reverse=True)
    return cases[: config.budgets.max_case_candidates]


def _append_case(
    cases: list[CaseCandidate],
    seen_keys: set[tuple[str, tuple[str, ...]]],
    case: CaseCandidate | None,
) -> None:
    if case is None or not case.member_transaction_ids:
        return
    signature = (case.case_type, tuple(sorted(case.member_transaction_ids)))
    if signature in seen_keys:
        return
    seen_keys.add(signature)
    cases.append(case)


def _recipient_campaign_case(
    store: FeatureStore,
    seeds: dict[str, CandidateSeed],
    config: AppConfig,
    recipient_key: str,
    tx_ids: list[str],
) -> CaseCandidate | None:
    unique_seed_ids = _ranked_ids(tx_ids, seeds)
    if len(unique_seed_ids) < 2:
        return None
    field_name, field_value = recipient_key.split("::", 1)
    if field_name == "recipient_iban":
        history = [tx for tx in store.transactions if tx.recipient_iban == field_value]
    else:
        history = store.recipient_histories.get(field_value, [])
    if len(history) < 2:
        return None
    ranked_history = _ranked_transactions(history, seeds)
    member_ids = [tx.transaction_id for tx in ranked_history[: config.budgets.max_case_members]]
    distinct_senders = len({tx.sender_id for tx in history if tx.sender_id})
    distinct_owners = len({tx.owner_key for tx in history if tx.owner_key})
    if distinct_senders < 2 and distinct_owners < 2:
        return None

    anchor_ids = unique_seed_ids[: min(4, len(unique_seed_ids))]
    amounts = [tx.amount for tx in history]
    seed_scores = [seeds[tx_id].score for tx_id in unique_seed_ids if tx_id in seeds]
    priority = min(1.0, max(seed_scores or [0.0]) + min(0.22, 0.04 * distinct_senders) + min(0.12, 0.03 * distinct_owners))
    economic = max(seeds[tx_id].economic_severity for tx_id in unique_seed_ids if tx_id in seeds)
    return _build_case(
        store=store,
        seeds=seeds,
        case_id=f"recipient_campaign::{recipient_key}",
        case_type="recipient_campaign",
        anchor_ids=anchor_ids,
        member_ids=member_ids,
        priority=priority,
        economic_severity=economic,
        shared_entities=[f"{field_name}:{field_value}"],
        summary={
            "recipient_key": recipient_key,
            "transaction_count": len(history),
            "distinct_senders": distinct_senders,
            "distinct_owner_keys": distinct_owners,
            "avg_amount": round(sum(amounts) / max(1, len(amounts)), 2),
            "top_seed_scores": seed_scores[:6],
        },
    )


def _owner_behavior_case(
    store: FeatureStore,
    seeds: dict[str, CandidateSeed],
    config: AppConfig,
    owner_key: str,
    tx_ids: list[str],
) -> CaseCandidate | None:
    unique_seed_ids = _ranked_ids(tx_ids, seeds)
    if not unique_seed_ids:
        return None
    history = store.owner_histories.get(owner_key, [])
    if len(history) < 2 and max((seeds[tx_id].economic_severity for tx_id in unique_seed_ids), default=0.0) < 0.7:
        return None
    ranked_history = _ranked_transactions(history or [store.transactions_by_id[tx_id] for tx_id in unique_seed_ids], seeds)
    member_ids = [tx.transaction_id for tx in ranked_history[: config.budgets.max_case_members]]
    distinct_recipients = len({tx.recipient_id for tx in history if tx.recipient_id})
    payment_methods = sorted({tx.payment_method for tx in history if tx.payment_method})[:4]
    priority = min(
        1.0,
        max((seeds[tx_id].score for tx_id in unique_seed_ids), default=0.0)
        + min(0.18, 0.04 * len(unique_seed_ids))
        + (0.08 if distinct_recipients >= 3 else 0.0),
    )
    economic = max((seeds[tx_id].economic_severity for tx_id in unique_seed_ids), default=0.0)
    return _build_case(
        store=store,
        seeds=seeds,
        case_id=f"owner_behavior::{owner_key}",
        case_type="owner_behavior_shift",
        anchor_ids=unique_seed_ids[: min(4, len(unique_seed_ids))],
        member_ids=member_ids,
        priority=priority,
        economic_severity=economic,
        shared_entities=[f"owner_key:{owner_key}"],
        summary={
            "owner_key": owner_key,
            "seed_count": len(unique_seed_ids),
            "history_count": len(history),
            "distinct_recipients": distinct_recipients,
            "payment_methods": payment_methods,
        },
    )


def _social_engineering_case(
    store: FeatureStore,
    seeds: dict[str, CandidateSeed],
    config: AppConfig,
    owner_key: str,
    tx_ids: list[str],
) -> CaseCandidate | None:
    unique_seed_ids = _ranked_ids(tx_ids, seeds)
    if not unique_seed_ids:
        return None
    txs = [store.transactions_by_id[tx_id] for tx_id in unique_seed_ids if tx_id in store.transactions_by_id]
    linked_messages = []
    for tx in txs:
        linked_messages.extend(
            store.linked_messages_for(
                tx,
                lookback_hours=config.linking.message_lookback_hours,
                max_items=config.budgets.max_messages_per_case,
            )
        )
    dedup_messages = {item.message_id: item for item in linked_messages}.values()
    risky_messages = [item for item in dedup_messages if item.risk_score >= config.seeds.message_risk_floor]
    if not risky_messages:
        return None
    priority = min(
        1.0,
        max((seeds[tx_id].score for tx_id in unique_seed_ids), default=0.0)
        + min(0.22, 0.06 * len(risky_messages)),
    )
    economic = max((seeds[tx_id].economic_severity for tx_id in unique_seed_ids), default=0.0)
    return _build_case(
        store=store,
        seeds=seeds,
        case_id=f"social_engineering::{owner_key}",
        case_type="victim_social_engineering",
        anchor_ids=unique_seed_ids[: min(4, len(unique_seed_ids))],
        member_ids=unique_seed_ids[: config.budgets.max_case_members],
        priority=priority,
        economic_severity=economic,
        shared_entities=[f"owner_key:{owner_key}"] + [message.sender_domain for message in risky_messages if message.sender_domain][:4],
        summary={
            "owner_key": owner_key,
            "linked_message_count": len(risky_messages),
            "top_message_risk": max((item.risk_score for item in risky_messages), default=0.0),
            "channels": sorted({item.channel for item in risky_messages}),
        },
    )


def _geo_case(
    store: FeatureStore,
    seeds: dict[str, CandidateSeed],
    config: AppConfig,
    owner_key: str,
    tx_ids: list[str],
) -> CaseCandidate | None:
    unique_seed_ids = _ranked_ids(tx_ids, seeds)
    if not unique_seed_ids:
        return None
    txs = [store.transactions_by_id[tx_id] for tx_id in unique_seed_ids if tx_id in store.transactions_by_id]
    cities = {store.geo_behavior_summary(tx)["transaction_city"] for tx in txs}
    cities.discard("")
    if not cities:
        return None
    priority = min(
        1.0,
        max((seeds[tx_id].score for tx_id in unique_seed_ids), default=0.0)
        + min(0.18, 0.05 * len(cities)),
    )
    economic = max((seeds[tx_id].economic_severity for tx_id in unique_seed_ids), default=0.0)
    return _build_case(
        store=store,
        seeds=seeds,
        case_id=f"geo_behavior::{owner_key}",
        case_type="geo_behavior_case",
        anchor_ids=unique_seed_ids[: min(4, len(unique_seed_ids))],
        member_ids=unique_seed_ids[: config.budgets.max_case_members],
        priority=priority,
        economic_severity=economic,
        shared_entities=[f"owner_key:{owner_key}"] + [f"city:{city}" for city in sorted(cities)[:4]],
        summary={
            "owner_key": owner_key,
            "distinct_cities": sorted(cities),
            "seed_count": len(unique_seed_ids),
        },
    )


def _graph_case(
    store: FeatureStore,
    seeds: dict[str, CandidateSeed],
    config: AppConfig,
    transaction_id: str,
) -> CaseCandidate | None:
    tx = store.transactions_by_id.get(transaction_id)
    if tx is None:
        return None
    neighbors = store.graph.expand(
        tx,
        max_size=max(4, config.budgets.max_case_members - 1),
        time_window_hours=config.budgets.max_cluster_time_window_hours,
    )
    if len(neighbors) < 2:
        return None
    member_ids = [transaction_id] + [neighbor.transaction_id for neighbor in neighbors[: config.budgets.max_case_members - 1]]
    seed_ids = [tx_id for tx_id in member_ids if seeds.get(tx_id) and seeds[tx_id].selected]
    if len(seed_ids) < 2:
        return None
    priority = min(
        1.0,
        max((seeds[tx_id].score for tx_id in seed_ids), default=0.0)
        + min(0.16, 0.03 * len(neighbors)),
    )
    economic = max((seeds[tx_id].economic_severity for tx_id in seed_ids), default=0.0)
    shared_entities = []
    for neighbor in neighbors[:4]:
        shared_entities.extend(neighbor.shared_entities[:2])
    return _build_case(
        store=store,
        seeds=seeds,
        case_id=f"graph_cluster::{transaction_id}",
        case_type="graph_cluster",
        anchor_ids=seed_ids[: min(3, len(seed_ids))] or [transaction_id],
        member_ids=member_ids,
        priority=priority,
        economic_severity=economic,
        shared_entities=list(dict.fromkeys(shared_entities))[:8],
        summary={
            "anchor_transaction_id": transaction_id,
            "neighbor_count": len(neighbors),
            "seed_member_count": len(seed_ids),
        },
    )


def _high_value_case(
    store: FeatureStore,
    seeds: dict[str, CandidateSeed],
    transaction_id: str,
) -> CaseCandidate | None:
    tx = store.transactions_by_id.get(transaction_id)
    seed = seeds.get(transaction_id)
    if tx is None or seed is None:
        return None
    return _build_case(
        store=store,
        seeds=seeds,
        case_id=f"high_value::{transaction_id}",
        case_type="high_value_anomaly",
        anchor_ids=[transaction_id],
        member_ids=[transaction_id],
        priority=min(1.0, seed.score + 0.12 * seed.economic_severity),
        economic_severity=seed.economic_severity,
        shared_entities=[f"transaction_id:{transaction_id}"],
        summary={
            "transaction_id": transaction_id,
            "amount": tx.amount,
            "economic_severity": seed.economic_severity,
            "seed_reasons": seed.reasons,
        },
    )


def _build_case(
    *,
    store: FeatureStore,
    seeds: dict[str, CandidateSeed],
    case_id: str,
    case_type: str,
    anchor_ids: list[str],
    member_ids: list[str],
    priority: float,
    economic_severity: float,
    shared_entities: list[str],
    summary: dict[str, object],
) -> CaseCandidate:
    seed_ids = [tx_id for tx_id in member_ids if tx_id in seeds]
    timestamps = [
        store.transactions_by_id[tx_id].timestamp
        for tx_id in member_ids
        if tx_id in store.transactions_by_id and store.transactions_by_id[tx_id].timestamp is not None
    ]
    owner_keys = sorted(
        {
            store.transactions_by_id[tx_id].owner_key
            for tx_id in member_ids
            if tx_id in store.transactions_by_id and store.transactions_by_id[tx_id].owner_key
        }
    )
    modality_coverage = _modality_coverage(store, member_ids)
    return CaseCandidate(
        case_id=case_id,
        case_type=case_type,
        anchor_transaction_ids=anchor_ids,
        member_transaction_ids=member_ids,
        seed_transaction_ids=seed_ids,
        owner_keys=owner_keys,
        shared_entities=list(dict.fromkeys(entity for entity in shared_entities if entity))[:12],
        modality_coverage=modality_coverage,
        priority_hint=min(1.0, priority),
        economic_severity=min(1.0, economic_severity),
        earliest_timestamp=min(timestamps) if timestamps else None,
        latest_timestamp=max(timestamps) if timestamps else None,
        summary=summary,
    )


def _ranked_ids(tx_ids: list[str], seeds: dict[str, CandidateSeed]) -> list[str]:
    unique_ids = list(dict.fromkeys(tx_ids))
    unique_ids.sort(
        key=lambda tx_id: (
            seeds.get(tx_id).score if tx_id in seeds else 0.0,
            seeds.get(tx_id).economic_severity if tx_id in seeds else 0.0,
        ),
        reverse=True,
    )
    return unique_ids


def _ranked_transactions(transactions, seeds: dict[str, CandidateSeed]):
    dedup = {tx.transaction_id: tx for tx in transactions}
    ranked = list(dedup.values())
    ranked.sort(
        key=lambda tx: (
            seeds.get(tx.transaction_id).score if tx.transaction_id in seeds else 0.0,
            seeds.get(tx.transaction_id).economic_severity if tx.transaction_id in seeds else 0.0,
            tx.amount,
        ),
        reverse=True,
    )
    return ranked


def _modality_coverage(store: FeatureStore, member_ids: list[str]) -> list[str]:
    coverage = {"transactions"}
    for tx_id in member_ids:
        tx = store.transactions_by_id.get(tx_id)
        if tx is None:
            continue
        if store.linked_messages_for(tx, lookback_hours=168, max_items=1):
            coverage.add("messages")
        if store.linked_audio_for(tx, lookback_hours=12, max_items=1):
            coverage.add("audio")
        geo = store.geo_behavior_summary(tx)
        if geo["transaction_city"] or geo["recent_location_cities"]:
            coverage.add("locations")
    return sorted(coverage)
