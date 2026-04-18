from __future__ import annotations

import argparse
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from echo_fraud_agents.case_builder import build_case_candidates
from echo_fraud_agents.candidate_generator import generate_high_recall_seeds
from echo_fraud_agents.cluster import ClusterExpander, ClusterSpecialistAgent
from echo_fraud_agents.config import AppConfig
from echo_fraud_agents.data_loading import DatasetLoader
from echo_fraud_agents.feature_store import build_feature_store
from echo_fraud_agents.judge import JudgeAgent
from echo_fraud_agents.llm_client import OpenRouterLLMClient
from echo_fraud_agents.memory import FraudMemory
from echo_fraud_agents.models import CaseReview, EvaluatedTransaction, RoutingDecision
from echo_fraud_agents.normalization import normalize_dataset
from echo_fraud_agents.output import OutputWriter
from echo_fraud_agents.specialists import AudioSpecialist, CommunicationSpecialist, GeoBehaviorSpecialist, TransactionPatternSpecialist
from echo_fraud_agents.supervisor import CaseDiscoveryAgent, TriageAgent
from echo_fraud_agents.tracing import LangfuseRuntime
from echo_fraud_agents.utils import chunked


class FraudPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.loader = DatasetLoader(scan_nested_archives=config.scan_nested_archives, max_archive_depth=config.max_archive_depth)
        self.tracing = LangfuseRuntime(config)
        self.llm_client = OpenRouterLLMClient(config, self.tracing)
        self.case_discovery = CaseDiscoveryAgent(config, self.llm_client)
        self.triage = TriageAgent(config, self.llm_client)
        self.communication = CommunicationSpecialist(config, self.llm_client)
        self.transaction_pattern = TransactionPatternSpecialist(config, self.llm_client)
        self.geo_behavior = GeoBehaviorSpecialist(config, self.llm_client)
        self.audio = AudioSpecialist(config, self.llm_client)
        self.cluster_expander = ClusterExpander(config)
        self.cluster_specialist = ClusterSpecialistAgent(config, self.llm_client)
        self.judge = JudgeAgent(config, self.llm_client)
        self.output_writer = OutputWriter()

    def run(
        self,
        *,
        input_path: Path,
        output_root: Path | None = None,
        dataset_filter: str | None = None,
    ) -> list[Path]:
        if not self.llm_client.available:
            raise RuntimeError(f"LLM runtime is not available: {self.llm_client.status()['status_reason']}")
        bundles = self.loader.discover(input_path)
        if dataset_filter:
            filter_text = dataset_filter.strip().lower()
            bundles = [bundle for bundle in bundles if filter_text in bundle.slug.lower() or filter_text in bundle.name.lower()]
        if not bundles:
            raise RuntimeError(f"No dataset bundles discovered in {input_path}")
        resolved_output_root = (output_root or self.config.output_root).resolve()
        logger = _configure_logging(self.config.log_level, resolved_output_root / "pipeline.log")
        outputs: list[Path] = []

        for bundle in bundles:
            logger.info("Processing bundle %s", bundle.name)
            dataset = normalize_dataset(bundle)
            store = build_feature_store(dataset, self.config)
            seeds = generate_high_recall_seeds(store, self.config)
            case_candidates = build_case_candidates(store, seeds, self.config)
            session_id = self.tracing.build_session_id()
            memory = FraudMemory()
            evaluated: dict[str, EvaluatedTransaction] = {}
            case_reviews: list[CaseReview] = []
            counters = Counter()
            route_counts = Counter()
            case_type_counts = Counter()
            model_usage = Counter()

            ordered_cases = sorted(
                case_candidates,
                key=lambda item: (item.earliest_timestamp or datetime.min, -item.priority_hint, -item.economic_severity),
            )
            for batch in chunked(ordered_cases, self.config.budgets.batch_size):
                for case in batch:
                    case_context = store.case_context(
                        case,
                        seeds,
                        max_transaction_items=min(12, self.config.budgets.max_case_members),
                        max_message_items=self.config.budgets.max_messages_per_case,
                        max_audio_items=self.config.budgets.max_audio_per_case,
                        message_lookback_hours=self.config.linking.message_lookback_hours,
                        audio_lookback_hours=self.config.linking.audio_lookback_hours,
                    )
                    discovery = None
                    if counters["case_discovery"] < self.config.budgets.max_case_discovery_calls:
                        discovery = self.case_discovery.discover(
                            session_id=session_id,
                            payload={"case": case_context, "memory_summary": memory.summary()},
                        )
                        counters["case_discovery"] += 1
                        if discovery and discovery.model:
                            model_usage[discovery.model] += 1
                            case_type_counts[discovery.case_type] += 1

                    triage = None
                    if counters["triage"] < self.config.budgets.max_supervisor_calls:
                        triage = self.triage.route(
                            session_id=session_id,
                            payload={
                                "case_type": discovery.case_type if discovery else case.case_type,
                                "case": case_context,
                                "discovery": _discovery_payload(discovery),
                                "memory_summary": memory.summary(),
                            },
                        )
                        counters["triage"] += 1
                        if triage and triage.model:
                            model_usage[triage.model] += 1

                    triage = _effective_triage(case, discovery, triage)
                    route_counts[triage.route] += 1
                    routes = _merged_routes(discovery, triage)
                    if triage.route == "ignore" and not triage.mandatory_tribunal and all(route == "ignore" for route in routes):
                        case_reviews.append(CaseReview(case=case, discovery=discovery, routing=triage, specialists=[], cluster=None, tribunal=None, second_pass=None))
                        continue

                    specialists = []
                    linked_messages = case_context["linked_messages"]
                    linked_audio = case_context["linked_audio"]
                    if "communication" in routes and linked_messages and counters["communication"] < self.config.budgets.max_communication_calls:
                        opinion = self.communication.review(
                            session_id=session_id,
                            payload={"case": case_context, "discovery": _discovery_payload(discovery), "triage": _triage_payload(triage), "memory_summary": memory.summary()},
                        )
                        counters["communication"] += 1
                        if opinion:
                            specialists.append(opinion)
                            if opinion.model:
                                model_usage[opinion.model] += 1
                    if "transaction_pattern" in routes and counters["transaction_pattern"] < self.config.budgets.max_transaction_pattern_calls:
                        opinion = self.transaction_pattern.review(
                            session_id=session_id,
                            payload={"case": case_context, "discovery": _discovery_payload(discovery), "triage": _triage_payload(triage), "memory_summary": memory.summary()},
                        )
                        counters["transaction_pattern"] += 1
                        if opinion:
                            specialists.append(opinion)
                            if opinion.model:
                                model_usage[opinion.model] += 1
                    if "geo_profile" in routes and counters["geo_behavior"] < self.config.budgets.max_geo_behavior_calls:
                        opinion = self.geo_behavior.review(
                            session_id=session_id,
                            payload={"case": case_context, "discovery": _discovery_payload(discovery), "triage": _triage_payload(triage), "memory_summary": memory.summary()},
                        )
                        counters["geo_behavior"] += 1
                        if opinion:
                            specialists.append(opinion)
                            if opinion.model:
                                model_usage[opinion.model] += 1
                    if self.config.runtime.audio_enabled and "audio" in routes and linked_audio and counters["audio"] < self.config.budgets.max_audio_calls:
                        opinion = self.audio.review(
                            session_id=session_id,
                            payload={"case": case_context, "discovery": _discovery_payload(discovery), "triage": _triage_payload(triage), "memory_summary": memory.summary()},
                        )
                        counters["audio"] += 1
                        if opinion:
                            specialists.append(opinion)
                            if opinion.model:
                                model_usage[opinion.model] += 1

                    cluster_decision = None
                    if ("cluster" in routes or any(opinion.expand_neighbors for opinion in specialists)) and counters["cluster"] < self.config.budgets.max_cluster_calls:
                        anchor_ids = case.anchor_transaction_ids or case.member_transaction_ids
                        anchors = [store.transactions_by_id.get(tx_id) for tx_id in anchor_ids if tx_id in store.transactions_by_id]
                        if anchors:
                            cluster_members = self.cluster_expander.expand(store, anchors)
                            if cluster_members:
                                cluster_decision = self.cluster_specialist.review(
                                    session_id=session_id,
                                    payload={
                                        "case": case_context,
                                        "discovery": _discovery_payload(discovery),
                                        "triage": _triage_payload(triage),
                                        "specialists": [_specialist_payload(opinion) for opinion in specialists],
                                        "cluster_members": cluster_members,
                                        "memory_summary": memory.summary(),
                                    },
                                )
                                counters["cluster"] += 1
                                if cluster_decision and cluster_decision.model:
                                    model_usage[cluster_decision.model] += 1

                    tribunal = None
                    if ("judge" in routes or triage.mandatory_tribunal) and counters["judge"] < self.config.budgets.max_judge_calls:
                        tribunal = self.judge.review(
                            session_id=session_id,
                            payload={
                                "case": case_context,
                                "discovery": _discovery_payload(discovery),
                                "triage": _triage_payload(triage),
                                "specialists": [_specialist_payload(opinion) for opinion in specialists],
                                "cluster": _cluster_payload(cluster_decision),
                                "memory_summary": memory.summary(),
                            },
                        )
                        counters["judge"] += 1
                        if tribunal and tribunal.model:
                            model_usage[tribunal.model] += 1

                    second_pass = None
                    if tribunal is not None and _needs_second_pass(case, triage, specialists, cluster_decision, tribunal) and counters["second_pass"] < self.config.budgets.max_second_pass_calls:
                        second_pass = self.judge.review_second_pass(
                            session_id=session_id,
                            payload={
                                "case": case_context,
                                "discovery": _discovery_payload(discovery),
                                "triage": _triage_payload(triage),
                                "specialists": [_specialist_payload(opinion) for opinion in specialists],
                                "cluster": _cluster_payload(cluster_decision),
                                "first_tribunal": _judge_payload(tribunal),
                                "memory_summary": memory.summary(),
                            },
                        )
                        counters["second_pass"] += 1
                        if second_pass and second_pass.model:
                            model_usage[second_pass.model] += 1

                    final_tribunal = second_pass or tribunal
                    selected_ids, borderline_ids, excluded_ids = _materialize_case_outcome(case=case, store=store, cluster=cluster_decision, tribunal=final_tribunal)
                    case_review = CaseReview(
                        case=case,
                        discovery=discovery,
                        routing=triage,
                        specialists=specialists,
                        cluster=cluster_decision,
                        tribunal=tribunal,
                        second_pass=second_pass,
                        selected_ids=selected_ids,
                        borderline_ids=borderline_ids,
                        excluded_ids=excluded_ids,
                    )
                    case_reviews.append(case_review)
                    _apply_case_review(store=store, seeds=seeds, evaluated=evaluated, case_review=case_review)
                    if final_tribunal and selected_ids:
                        memory.remember_case(case=case, tribunal=final_tribunal, specialists=specialists, cluster=cluster_decision)

            selected_ids, guardrail_details = _apply_guardrails(evaluated=evaluated, total_transactions=len(store.transactions), config=self.config)
            if counters["case_discovery"] > 0 and not route_counts and self.llm_client.last_error:
                logger.warning("All case triage calls failed for %s: %s", bundle.name, self.llm_client.last_error)

            output_dir = resolved_output_root / dataset.manifest.slug
            run_summary = {
                "dataset": {
                    "name": dataset.manifest.name,
                    "slug": dataset.manifest.slug,
                    "modalities": dataset.manifest.modalities,
                    "record_counts": dataset.manifest.record_counts,
                },
                "session_id": session_id,
                "selected_count": len(selected_ids),
                "seed_count": sum(1 for seed in seeds.values() if seed.selected),
                "case_candidate_count": len(case_candidates),
                "reviewed_case_count": len(case_reviews),
                "route_counts": dict(route_counts),
                "case_type_counts": dict(case_type_counts),
                "budget_usage": dict(counters),
                "model_usage": dict(model_usage),
                "guardrails": guardrail_details,
                "llm_status": self.llm_client.status(),
                "langfuse_status": self.tracing.status(session_id),
            }
            diagnostics = {
                "memory_summary": memory.summary(),
                "seed_diagnostics": {
                    tx_id: {"score": seed.score, "reasons": seed.reasons, "signal_breakdown": seed.signal_breakdown, "selected": seed.selected}
                    for tx_id, seed in list(seeds.items())[: min(250, len(seeds))]
                },
                "case_candidates": [
                    {
                        "case_id": case.case_id,
                        "case_type": case.case_type,
                        "priority_hint": case.priority_hint,
                        "economic_severity": case.economic_severity,
                        "anchor_transaction_ids": case.anchor_transaction_ids,
                        "member_transaction_ids": case.member_transaction_ids,
                        "shared_entities": case.shared_entities,
                        "summary": case.summary,
                    }
                    for case in case_candidates[: min(120, len(case_candidates))]
                ],
            }
            session_artifact = self.tracing.status(session_id)
            self.output_writer.write_bundle_outputs(
                output_dir=output_dir,
                selected_ids=selected_ids,
                seeds=seeds,
                evaluated=evaluated,
                case_reviews=case_reviews,
                run_summary=run_summary,
                diagnostics=diagnostics,
                session_artifact=session_artifact,
            )
            outputs.append(output_dir)
            logger.info("Bundle %s complete with %s selected transactions", bundle.name, len(selected_ids))

        self.tracing.flush()
        return outputs


def _effective_triage(case, discovery, triage):
    if triage is None:
        return RoutingDecision(
            route="judge",
            routes=["judge"],
            priority=max(case.priority_hint, discovery.priority if discovery else 0.0),
            reason="triage_unavailable_escalated_to_tribunal",
            mandatory_tribunal=True,
            escalate_on_borderline=True,
            case_type=discovery.case_type if discovery else case.case_type,
        )
    if discovery is not None:
        triage.case_type = discovery.case_type
        triage.priority = max(triage.priority, discovery.priority)
        triage.mandatory_tribunal = triage.mandatory_tribunal or discovery.mandatory_tribunal
    else:
        triage.case_type = triage.case_type or case.case_type
        triage.priority = max(triage.priority, case.priority_hint)
    return triage


def _merged_routes(discovery, triage) -> list[str]:
    routes = []
    for source in (discovery.required_routes if discovery else [], triage.routes if triage else []):
        for route in source:
            if route not in routes:
                routes.append(route)
    if triage and triage.mandatory_tribunal and "judge" not in routes:
        routes.append("judge")
    return routes or ["judge"]


def _needs_second_pass(case, triage, specialists, cluster, tribunal) -> bool:
    if tribunal.verdict == "borderline":
        return True
    if triage.mandatory_tribunal and tribunal.final_score < 0.62:
        return True
    if case.economic_severity >= 0.8 and tribunal.final_score < 0.7:
        return True
    scores = [opinion.score for opinion in specialists]
    if scores and max(scores) >= 0.7 and min(scores) <= 0.25:
        return True
    if cluster and cluster.confidence >= 0.7 and tribunal.verdict == "exclude":
        return True
    return False


def _materialize_case_outcome(*, case, store, cluster, tribunal):
    if tribunal is None:
        return [], [], []
    include_ids = [tx_id for tx_id in tribunal.include_ids if tx_id in store.transactions_by_id]
    borderline_ids = [tx_id for tx_id in tribunal.borderline_ids if tx_id in store.transactions_by_id]
    exclude_ids = [tx_id for tx_id in tribunal.exclude_ids if tx_id in store.transactions_by_id]
    if cluster is not None and tribunal.verdict == "include":
        for tx_id in cluster.include_ids:
            if tx_id in store.transactions_by_id and tx_id not in include_ids:
                include_ids.append(tx_id)
    if tribunal.verdict == "include" and not include_ids:
        include_ids = [tx_id for tx_id in case.anchor_transaction_ids if tx_id in store.transactions_by_id]
    if tribunal.verdict == "borderline" and not borderline_ids:
        borderline_ids = [tx_id for tx_id in case.anchor_transaction_ids if tx_id in store.transactions_by_id]
    if tribunal.verdict == "exclude" and not exclude_ids:
        exclude_ids = [tx_id for tx_id in case.anchor_transaction_ids if tx_id in store.transactions_by_id]
    return list(dict.fromkeys(include_ids)), list(dict.fromkeys(borderline_ids)), list(dict.fromkeys(exclude_ids))


def _apply_case_review(*, store, seeds, evaluated, case_review) -> None:
    final_tribunal = case_review.second_pass or case_review.tribunal
    involved_ids = list(dict.fromkeys([*case_review.case.anchor_transaction_ids, *case_review.selected_ids, *case_review.borderline_ids, *case_review.excluded_ids]))
    for tx_id in involved_ids:
        tx = store.transactions_by_id.get(tx_id)
        if tx is None:
            continue
        item = evaluated.get(tx_id)
        if item is None:
            item = EvaluatedTransaction(transaction=tx, candidate_seed=seeds.get(tx_id))
            evaluated[tx_id] = item
        if case_review.case.case_id not in item.case_ids:
            item.case_ids.append(case_review.case.case_id)
        if case_review.case.case_type not in item.case_types:
            item.case_types.append(case_review.case.case_type)
        if case_review.routing is not None:
            item.routing = case_review.routing
        for opinion in case_review.specialists:
            if opinion not in item.specialists:
                item.specialists.append(opinion)
        if case_review.cluster is not None:
            item.cluster = case_review.cluster
        if final_tribunal is not None:
            item.judge = final_tribunal
            if final_tribunal.fraud_mechanism and final_tribunal.fraud_mechanism not in item.fraud_mechanisms:
                item.fraud_mechanisms.append(final_tribunal.fraud_mechanism)
            if final_tribunal.rationale and final_tribunal.rationale not in item.rationales:
                item.rationales.append(final_tribunal.rationale)
            item.tribunal_scores.append(final_tribunal.final_score)
        if tx_id in case_review.selected_ids:
            item.selected = True
        if case_review.cluster and tx_id in case_review.cluster.include_ids and tx_id not in case_review.case.anchor_transaction_ids:
            item.selected_via_cluster = True


def _apply_guardrails(*, evaluated: dict[str, EvaluatedTransaction], total_transactions: int, config: AppConfig) -> tuple[list[str], dict[str, Any]]:
    ranked = sorted(
        evaluated.values(),
        key=lambda item: (
            item.selected,
            item.judge.final_score if item.judge else 0.0,
            item.candidate_seed.score if item.candidate_seed else 0.0,
            max(item.tribunal_scores) if item.tribunal_scores else 0.0,
        ),
        reverse=True,
    )
    selected_ids = {item.transaction.transaction_id for item in ranked if item.selected}
    borderlines = [item for item in ranked if item.transaction.transaction_id not in selected_ids and item.judge is not None and item.judge.verdict == "borderline"]
    if len(selected_ids) < config.guardrails.min_selected_count:
        needed = min(config.guardrails.max_borderline_promotions, config.guardrails.min_selected_count - len(selected_ids))
        for item in borderlines[:needed]:
            item.selected = True
            selected_ids.add(item.transaction.transaction_id)
    if not selected_ids:
        fallback = [item for item in ranked if item.judge is not None][: config.guardrails.min_selected_count]
        for item in fallback:
            item.selected = True
            selected_ids.add(item.transaction.transaction_id)
    cap = max(config.guardrails.min_selected_count, min(config.guardrails.max_selected_count, int(total_transactions * config.guardrails.max_output_fraction) or config.guardrails.min_selected_count))
    if len(selected_ids) > cap:
        trimmed = sorted(
            [item for item in ranked if item.transaction.transaction_id in selected_ids],
            key=lambda item: (
                item.judge.final_score if item.judge else 0.0,
                item.candidate_seed.score if item.candidate_seed else 0.0,
                max(item.tribunal_scores) if item.tribunal_scores else 0.0,
            ),
            reverse=True,
        )[:cap]
        keep_ids = {item.transaction.transaction_id for item in trimmed}
        for item in ranked:
            item.selected = item.transaction.transaction_id in keep_ids
        selected_ids = keep_ids
    final_ids = sorted(
        selected_ids,
        key=lambda tx_id: (
            evaluated[tx_id].judge.final_score if tx_id in evaluated and evaluated[tx_id].judge else 0.0,
            evaluated[tx_id].candidate_seed.score if tx_id in evaluated and evaluated[tx_id].candidate_seed else 0.0,
            max(evaluated[tx_id].tribunal_scores) if tx_id in evaluated and evaluated[tx_id].tribunal_scores else 0.0,
        ),
        reverse=True,
    )
    return final_ids, {
        "final_count": len(final_ids),
        "max_cap": cap,
        "borderline_promotions": sum(1 for item in evaluated.values() if item.selected and item.judge is not None and item.judge.verdict == "borderline"),
    }


def _discovery_payload(discovery) -> dict[str, Any] | None:
    if discovery is None:
        return None
    return {
        "case_type": discovery.case_type,
        "priority": discovery.priority,
        "mandatory_tribunal": discovery.mandatory_tribunal,
        "required_routes": discovery.required_routes,
        "suspected_mechanism": discovery.suspected_mechanism,
        "reason": discovery.reason,
    }


def _triage_payload(triage) -> dict[str, Any] | None:
    if triage is None:
        return None
    return {
        "route": triage.route,
        "routes": triage.routes,
        "priority": triage.priority,
        "mandatory_tribunal": triage.mandatory_tribunal,
        "escalate_on_borderline": triage.escalate_on_borderline,
        "reason": triage.reason,
        "case_type": triage.case_type,
    }


def _specialist_payload(opinion) -> dict[str, Any]:
    return {
        "stage": opinion.stage,
        "score": opinion.score,
        "confidence": opinion.confidence,
        "label": opinion.label,
        "rationale": opinion.rationale,
        "suspicious_entities": opinion.suspicious_entities,
        "expand_neighbors": opinion.expand_neighbors,
    }


def _cluster_payload(cluster) -> dict[str, Any] | None:
    if cluster is None:
        return None
    return {
        "cluster_score": cluster.cluster_score,
        "confidence": cluster.confidence,
        "include_ids": cluster.include_ids,
        "suspicious_entities": cluster.suspicious_entities,
        "rationale": cluster.rationale,
        "fraud_mechanism": cluster.fraud_mechanism,
    }


def _judge_payload(judge) -> dict[str, Any] | None:
    if judge is None:
        return None
    return {
        "fraud_mechanism": judge.fraud_mechanism,
        "final_score": judge.final_score,
        "verdict": judge.verdict,
        "include_ids": judge.include_ids,
        "borderline_ids": judge.borderline_ids,
        "exclude_ids": judge.exclude_ids,
        "key_entities": judge.key_entities,
        "rationale": judge.rationale,
        "confidence": judge.confidence,
    }


def _configure_logging(level: str, log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("echo_fraud_agents")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Echo Fraud Agents pipeline.")
    parser.add_argument("--input", required=True, help="Dataset folder or zip archive.")
    parser.add_argument("--config", help="Optional TOML config override.")
    parser.add_argument("--output-root", help="Optional output directory override.")
    parser.add_argument("--dataset-filter", help="Optional bundle name or slug filter after discovery.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    root_dir = Path(__file__).resolve().parents[2]
    config = AppConfig.load(root_dir=root_dir, config_path=Path(args.config).resolve() if args.config else None)
    pipeline = FraudPipeline(config)
    pipeline.run(
        input_path=Path(args.input).resolve(),
        output_root=Path(args.output_root).resolve() if args.output_root else None,
        dataset_filter=args.dataset_filter,
    )
    return 0
