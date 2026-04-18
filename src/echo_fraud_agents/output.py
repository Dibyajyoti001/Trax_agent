from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from echo_fraud_agents.models import CandidateSeed, CaseReview, EvaluatedTransaction


class OutputWriter:
    def write_bundle_outputs(
        self,
        *,
        output_dir: Path,
        selected_ids: list[str],
        seeds: dict[str, CandidateSeed],
        evaluated: dict[str, EvaluatedTransaction],
        case_reviews: list[CaseReview],
        run_summary: dict[str, Any],
        diagnostics: dict[str, Any],
        session_artifact: dict[str, Any],
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ranked_rows = self._ranked_rows(seeds, evaluated)
        case_rows = self._case_rows(case_reviews)

        submission_path = output_dir / "submission.txt"
        submission_path.write_text("\n".join(selected_ids), encoding="ascii", errors="ignore")

        (output_dir / "run_summary.json").write_text(
            json.dumps(_jsonable(run_summary), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "langfuse_session.json").write_text(
            json.dumps(_jsonable(session_artifact), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "diagnostics.json").write_text(
            json.dumps(_jsonable(diagnostics), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "ranked_transactions.json").write_text(
            json.dumps(_jsonable(ranked_rows), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "case_reviews.json").write_text(
            json.dumps(_jsonable(case_rows), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._write_csv(output_dir / "ranked_transactions.csv", ranked_rows)
        self._write_csv(output_dir / "case_reviews.csv", case_rows)

    def _ranked_rows(
        self,
        seeds: dict[str, CandidateSeed],
        evaluated: dict[str, EvaluatedTransaction],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        transaction_ids = set(seeds) | set(evaluated)
        for transaction_id in transaction_ids:
            seed = seeds.get(transaction_id)
            item = evaluated.get(transaction_id)
            judge = item.judge if item else None
            cluster = item.cluster if item else None
            routes = item.routing.routes if item and item.routing else []
            rows.append(
                {
                    "transaction_id": transaction_id,
                    "selected": bool(item.selected) if item else False,
                    "selected_via_cluster": bool(item.selected_via_cluster) if item else False,
                    "candidate_score": seed.score if seed else 0.0,
                    "economic_severity": seed.economic_severity if seed else 0.0,
                    "seed_reasons": "; ".join(seed.reasons) if seed else "",
                    "case_ids": "; ".join(item.case_ids) if item else "",
                    "case_types": "; ".join(item.case_types) if item else "",
                    "route": item.routing.route if item and item.routing else "",
                    "routes": "; ".join(routes),
                    "tribunal_verdict": judge.verdict if judge else "",
                    "tribunal_score": judge.final_score if judge else 0.0,
                    "tribunal_confidence": judge.confidence if judge else 0.0,
                    "fraud_mechanisms": "; ".join(item.fraud_mechanisms) if item else "",
                    "tribunal_rationales": " | ".join(item.rationales[:3]) if item else "",
                    "specialists": "; ".join(
                        f"{op.stage}:{op.label}:{op.score:.2f}:{op.confidence:.2f}"
                        for op in (item.specialists if item else [])
                    ),
                    "cluster_score": cluster.cluster_score if cluster else 0.0,
                    "cluster_include_ids": "; ".join(cluster.include_ids) if cluster else "",
                }
            )
        rows.sort(
            key=lambda row: (
                row["selected"],
                row["tribunal_score"],
                row["candidate_score"],
                row["economic_severity"],
            ),
            reverse=True,
        )
        return rows

    def _case_rows(self, case_reviews: list[CaseReview]) -> list[dict[str, Any]]:
        rows = []
        for review in case_reviews:
            tribunal = review.second_pass or review.tribunal
            routing = review.routing
            discovery = review.discovery
            rows.append(
                {
                    "case_id": review.case.case_id,
                    "case_type": review.case.case_type,
                    "anchor_ids": "; ".join(review.case.anchor_transaction_ids),
                    "member_count": len(review.case.member_transaction_ids),
                    "priority_hint": review.case.priority_hint,
                    "economic_severity": review.case.economic_severity,
                    "discovered_case_type": discovery.case_type if discovery else "",
                    "discovery_priority": discovery.priority if discovery else 0.0,
                    "mandatory_tribunal": routing.mandatory_tribunal if routing else bool(discovery.mandatory_tribunal if discovery else False),
                    "routes": "; ".join(routing.routes) if routing else "",
                    "tribunal_verdict": tribunal.verdict if tribunal else "",
                    "tribunal_score": tribunal.final_score if tribunal else 0.0,
                    "tribunal_confidence": tribunal.confidence if tribunal else 0.0,
                    "fraud_mechanism": tribunal.fraud_mechanism if tribunal else "",
                    "selected_ids": "; ".join(review.selected_ids),
                    "borderline_ids": "; ".join(review.borderline_ids),
                    "excluded_ids": "; ".join(review.excluded_ids),
                }
            )
        rows.sort(key=lambda row: (row["tribunal_score"], row["priority_hint"]), reverse=True)
        return rows

    def _write_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        fieldnames = list(rows[0].keys()) if rows else ["id"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
