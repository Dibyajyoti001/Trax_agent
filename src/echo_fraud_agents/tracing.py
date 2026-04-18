from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import ulid

from echo_fraud_agents.config import AppConfig
from echo_fraud_agents.utils import slugify


@dataclass(slots=True)
class _NullGeneration:
    trace_id: str | None = None

    def update(self, **_: Any) -> "_NullGeneration":
        return self

    def update_trace(self, **_: Any) -> "_NullGeneration":
        return self

    def end(self, **_: Any) -> "_NullGeneration":
        return self


class LangfuseRuntime:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = None
        self.status_reason = "disabled"
        self.trace_ids: dict[str, str] = {}
        try:
            from langfuse import Langfuse
        except ImportError:
            self.status_reason = "langfuse_not_installed"
            return

        if not (
            config.runtime.langfuse_public_key
            and config.runtime.langfuse_secret_key
            and config.runtime.langfuse_host
        ):
            self.status_reason = "missing_langfuse_credentials"
            return

        self.client = Langfuse(
            public_key=config.runtime.langfuse_public_key,
            secret_key=config.runtime.langfuse_secret_key,
            host=config.runtime.langfuse_host,
        )
        try:
            self.client.auth_check()
        except Exception:
            self.client = None
            self.status_reason = "langfuse_auth_failed"
            return
        self.status_reason = "ready"

    @property
    def available(self) -> bool:
        return self.client is not None

    def build_session_id(self) -> str:
        return f"{slugify(self.config.runtime.team_name)}-{str(ulid.new()).lower()}"

    def status(self, session_id: str | None = None) -> dict[str, Any]:
        trace_id = self.trace_ids.get(session_id or "", "")
        trace_url = None
        if self.client is not None and trace_id:
            try:
                trace_url = self.client.get_trace_url(trace_id)
            except Exception:
                trace_url = None
        return {
            "available": self.available,
            "status_reason": self.status_reason,
            "host": self.config.runtime.langfuse_host,
            "session_id": session_id,
            "trace_id": trace_id or None,
            "trace_url": trace_url,
        }

    def start_generation(
        self,
        *,
        session_id: str,
        stage: str,
        model: str,
        input_payload: Any,
        metadata: dict[str, Any] | None = None,
    ):
        if self.client is None:
            return _NullGeneration()
        generation = self.client.start_observation(
            as_type="generation",
            name=stage,
            model=model,
            input=input_payload,
            metadata=metadata or {},
        )
        generation.update_trace(
            name="echo_fraud_agents_run",
            session_id=session_id,
            tags=["reply-mirror", "openrouter", "llm-led"],
            metadata={"team_name": self.config.runtime.team_name},
        )
        self.trace_ids[session_id] = generation.trace_id
        return generation

    def flush(self) -> None:
        if self.client is not None:
            self.client.flush()


class LLMTraceHandle:
    def __init__(self, generation) -> None:
        self.generation = generation

    def set_output(self, output: Any) -> None:
        self.generation.update(output=output)

    def set_usage(self, *, input_tokens: int, output_tokens: int, total_tokens: int) -> None:
        self.generation.update(
            usage_details={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
        )

    def set_error(self, message: str) -> None:
        self.generation.update(level="ERROR", status_message=message)

    def end(self) -> None:
        self.generation.end()


@contextmanager
def trace_llm_call(
    *,
    runtime: LangfuseRuntime,
    session_id: str,
    stage: str,
    model: str,
    input_payload: Any,
    metadata: dict[str, Any] | None = None,
) -> Iterator[LLMTraceHandle]:
    handle = LLMTraceHandle(
        runtime.start_generation(
            session_id=session_id,
            stage=stage,
            model=model,
            input_payload=input_payload,
            metadata=metadata,
        )
    )
    try:
        yield handle
    except Exception as exc:
        handle.set_error(str(exc))
        handle.end()
        raise
    else:
        handle.end()
