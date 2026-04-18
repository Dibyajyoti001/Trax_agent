from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_dotenv(root_dir: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (root_dir / ".env", root_dir.parent / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(slots=True)
class RuntimeConfig:
    team_name: str
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    langfuse_host: str | None
    openrouter_api_keys: list[str]
    llm_enabled: bool
    audio_enabled: bool
    require_tracing_for_llm: bool
    request_timeout_seconds: int
    max_retries: int


@dataclass(slots=True)
class ModelsConfig:
    supervisor: list[str]
    communication: list[str]
    transaction_pattern: list[str]
    geo_behavior: list[str]
    cluster: list[str]
    audio: list[str]
    judge: list[str]


@dataclass(slots=True)
class BudgetsConfig:
    seed_fraction: float
    min_seed_count: int
    max_seed_count: int
    max_case_candidates: int
    max_case_members: int
    batch_size: int
    max_case_discovery_calls: int
    max_supervisor_calls: int
    max_communication_calls: int
    max_transaction_pattern_calls: int
    max_geo_behavior_calls: int
    max_audio_calls: int
    max_cluster_calls: int
    max_judge_calls: int
    max_second_pass_calls: int
    max_messages_per_case: int
    max_audio_per_case: int
    max_cluster_size: int
    max_cluster_time_window_hours: int


@dataclass(slots=True)
class LinkingConfig:
    message_lookback_hours: int
    audio_lookback_hours: int
    location_match_window_minutes: int


@dataclass(slots=True)
class SeedConfig:
    high_value_quantile: float
    high_amount_multiplier: float
    new_recipient_amount: float
    recipient_fan_in_threshold: int
    message_risk_floor: float
    graph_neighbor_floor: int


@dataclass(slots=True)
class TokenLimitConfig:
    supervisor: int
    specialist: int
    cluster: int
    judge: int


@dataclass(slots=True)
class GuardrailConfig:
    min_selected_count: int
    max_selected_count: int
    max_output_fraction: float
    max_borderline_promotions: int


@dataclass(slots=True)
class AppConfig:
    root_dir: Path
    output_root: Path
    cache_root: Path
    log_level: str
    scan_nested_archives: bool
    max_archive_depth: int
    runtime: RuntimeConfig
    models: ModelsConfig
    budgets: BudgetsConfig
    linking: LinkingConfig
    seeds: SeedConfig
    token_limits: TokenLimitConfig
    guardrails: GuardrailConfig

    @classmethod
    def load(cls, root_dir: Path, config_path: Path | None = None) -> "AppConfig":
        resolved_root = root_dir.resolve()
        _load_dotenv(resolved_root)
        default_path = resolved_root / "configs" / "default.toml"
        base = tomllib.loads(default_path.read_text(encoding="utf-8"))
        if config_path:
            override = tomllib.loads(config_path.read_text(encoding="utf-8"))
            data = _deep_merge(base, override)
        else:
            data = base

        runtime_cfg = data["runtime"]
        models_cfg = data["models"]

        keys_env = os.getenv("OPENROUTER_API_KEYS") or os.getenv("OPENROUTER_API_KEY")
        api_keys = [k.strip() for k in (keys_env or "").split(",") if k.strip()]

        runtime = RuntimeConfig(
            team_name=os.getenv("TEAM_NAME", runtime_cfg["team_name"]).strip() or "echo-fraud",
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            langfuse_host=os.getenv("LANGFUSE_HOST", runtime_cfg["langfuse_host"]),
            openrouter_api_keys=api_keys,
            llm_enabled=_env_bool("ECHO_LLM_ENABLED", bool(runtime_cfg["llm_enabled"])),
            audio_enabled=_env_bool("ECHO_AUDIO_ENABLED", bool(runtime_cfg["audio_enabled"])),
            require_tracing_for_llm=_env_bool(
                "ECHO_REQUIRE_TRACING_FOR_LLM",
                bool(runtime_cfg["require_tracing_for_llm"]),
            ),
            request_timeout_seconds=_env_int(
                "ECHO_REQUEST_TIMEOUT_SECONDS",
                int(runtime_cfg["request_timeout_seconds"]),
            ),
            max_retries=_env_int("ECHO_MAX_RETRIES", int(runtime_cfg["max_retries"])),
        )

        models = ModelsConfig(
            supervisor=_env_list("ECHO_MODELS_SUPERVISOR", list(models_cfg["supervisor"])),
            communication=_env_list("ECHO_MODELS_COMMUNICATION", list(models_cfg["communication"])),
            transaction_pattern=_env_list(
                "ECHO_MODELS_TRANSACTION_PATTERN",
                list(models_cfg["transaction_pattern"]),
            ),
            geo_behavior=_env_list("ECHO_MODELS_GEO_BEHAVIOR", list(models_cfg["geo_behavior"])),
            cluster=_env_list("ECHO_MODELS_CLUSTER", list(models_cfg["cluster"])),
            audio=_env_list("ECHO_MODELS_AUDIO", list(models_cfg["audio"])),
            judge=_env_list("ECHO_MODELS_JUDGE", list(models_cfg["judge"])),
        )

        return cls(
            root_dir=resolved_root,
            output_root=(resolved_root / data["paths"]["output_root"]).resolve(),
            cache_root=(resolved_root / data["paths"]["cache_root"]).resolve(),
            log_level=str(data["logging"]["level"]).upper(),
            scan_nested_archives=bool(data["ingestion"]["scan_nested_archives"]),
            max_archive_depth=int(data["ingestion"]["max_archive_depth"]),
            runtime=runtime,
            models=models,
            budgets=BudgetsConfig(**data["budgets"]),
            linking=LinkingConfig(**data["linking"]),
            seeds=SeedConfig(**data["seeds"]),
            token_limits=TokenLimitConfig(**data["token_limits"]),
            guardrails=GuardrailConfig(**data["guardrails"]),
        )
