from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from echo_fraud_agents.config import AppConfig
from echo_fraud_agents.tracing import LangfuseRuntime, trace_llm_call


def _parse_json(text: str) -> dict[str, Any]:
    content = (text or "").strip()
    if content.startswith("```"):
        lines = [line for line in content.splitlines() if not line.strip().startswith("```")]
        content = "\n".join(lines).strip()
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(content[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


def coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    aliases = {
        "low": 0.25,
        "medium": 0.55,
        "moderate": 0.55,
        "high": 0.85,
        "strong": 0.85,
        "critical": 0.98,
    }
    if text in aliases:
        return aliases[text]
    try:
        return float(text)
    except ValueError:
        return default


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "include", "expand"}:
        return True
    if text in {"false", "0", "no", "n", "exclude", "ignore"}:
        return False
    return default


def coerce_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not value:
        return []
    return [str(value).strip()]


class OpenRouterLLMClient:
    def __init__(self, config: AppConfig, tracing: LangfuseRuntime) -> None:
        self.config = config
        self.tracing = tracing
        self.clients = []
        self._client_idx = 0
        self.status_reason = "disabled"
        self.last_error = ""

        if not config.runtime.llm_enabled:
            self.status_reason = "llm_disabled"
            return
        if config.runtime.require_tracing_for_llm and not tracing.available:
            self.status_reason = "tracing_required_but_unavailable"
            return
        if not config.runtime.openrouter_api_keys:
            self.status_reason = "missing_openrouter_api_key"
            return

        for key in config.runtime.openrouter_api_keys:
            self.clients.append(OpenAI(
                api_key=key,
                base_url="https://openrouter.ai/api/v1",
                timeout=config.runtime.request_timeout_seconds,
                default_headers={
                    "HTTP-Referer": "https://challenges.reply.com",
                    "X-Title": "echo_fraud_agents",
                },
            ))
        self.status_reason = "ready"

    @property
    def available(self) -> bool:
        return len(self.clients) > 0

    def _next_client(self) -> OpenAI:
        client = self.clients[self._client_idx]
        self._client_idx = (self._client_idx + 1) % len(self.clients)
        return client

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "status_reason": self.status_reason,
            "last_error": self.last_error or None,
        }

    def call_role_json(
        self,
        *,
        session_id: str,
        stage: str,
        models: list[str],
        system_prompt: str,
        payload: dict[str, Any],
        max_tokens: int,
    ) -> dict[str, Any]:
        if not self.available:
            self.last_error = self.status_reason
            return {"_llm_failed": True, "error": self.status_reason}

        last_error = None
        for model in models:
            for _ in range(max(1, len(self.clients))):
                client = self._next_client()
                try:
                    with trace_llm_call(
                        runtime=self.tracing,
                        session_id=session_id,
                        stage=stage,
                        model=model,
                        input_payload=payload,
                        metadata={"stage": stage},
                    ) as tracer:
                        response = client.chat.completions.create(
                            model=model,
                            temperature=0.0,
                            max_tokens=max_tokens,
                            response_format={"type": "json_object"},
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                            ],
                        )
                    content = response.choices[0].message.content or "{}"
                    tracer.set_output(content)
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        tracer.set_usage(
                            input_tokens=getattr(usage, "prompt_tokens", 0),
                            output_tokens=getattr(usage, "completion_tokens", 0),
                            total_tokens=getattr(usage, "total_tokens", 0),
                        )

                    data = _parse_json(content)
                    data["_llm_failed"] = False
                    data["_llm_model"] = model
                    self.last_error = ""
                    return data
                except Exception as exc:
                    last_error = str(exc)
                    if "429" in last_error or "rate limit" in last_error.lower():
                        continue
                    break
        self.last_error = last_error or "all_models_failed"
        return {"_llm_failed": True, "_llm_model": None, "error": self.last_error}
