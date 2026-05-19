"""
Author: L. Saetta
Last update: 2026-05-19
License: MIT
Description: OpenTelemetry and Langfuse hook helpers for the RAG demo.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import TYPE_CHECKING, Any

from locus.hooks.provider import (
    AfterToolCallEvent,
    BeforeToolCallEvent,
    HookPriority,
    HookProvider,
)
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode

from demos.demo_rag.config import DemoConfig

if TYPE_CHECKING:
    from locus.core.state import AgentState

LOGGER = logging.getLogger(__name__)
_LANGFUSE_OTEL_CONFIGURED = False


def build_hooks(config: DemoConfig) -> list[Any]:
    """Build Locus hooks configured for the demo.

    Args:
        config: Demo configuration loaded from environment variables.

    Returns:
        Locus hook providers for the agent. Returns an empty list when
        Langfuse telemetry is disabled.
    """
    if not config.langfuse.enabled:
        return []

    configure_langfuse_otel(config)
    return [LangfuseTelemetryHook(service_name=config.langfuse.service_name)]


class LangfuseTelemetryHook(HookProvider):
    """OpenTelemetry hook that keeps Locus spans in one trace tree."""

    def __init__(
        self,
        service_name: str,
        tracer_name: str = "demos.demo_rag.telemetry",
    ) -> None:
        """Initialize the Langfuse telemetry hook.

        Args:
            service_name: OpenTelemetry service name to set on root spans.
            tracer_name: Name used to create the OpenTelemetry tracer.
        """
        self._service_name = service_name
        self._tracer = trace.get_tracer(tracer_name)
        self._invocation_spans: dict[str, tuple[Span, object]] = {}
        self._iteration_spans: dict[tuple[str, int], tuple[Span, object]] = {}
        self._tool_spans: dict[str, tuple[Span, object, float]] = {}

    @property
    def priority(self) -> int:
        """Return hook priority."""
        return HookPriority.OBSERVABILITY_MIN + 10

    @property
    def name(self) -> str:
        """Return hook name."""
        return "LangfuseTelemetryHook"

    def register_hooks(self) -> dict[str, bool]:
        """Register only the lifecycle hooks implemented here."""
        return {
            "on_before_invocation": True,
            "on_after_invocation": True,
            "on_before_tool_call": True,
            "on_after_tool_call": True,
            "on_iteration_start": True,
            "on_iteration_end": True,
            "on_before_model_call": False,
            "on_after_model_call": False,
        }

    async def on_before_invocation(
        self,
        prompt: str,
        state: "AgentState",
    ) -> "AgentState":
        """Start the root agent invocation span and attach it as current."""
        span = self._tracer.start_span(
            "agent.invocation",
            attributes={
                "locus.run_id": state.run_id,
                "locus.agent_id": state.agent_id or "",
                "locus.prompt_length": len(prompt),
                "locus.max_iterations": state.max_iterations,
                "service.name": self._service_name,
            },
        )
        token = otel_context.attach(trace.set_span_in_context(span))
        self._invocation_spans[state.run_id] = (span, token)
        return state

    async def on_after_invocation(
        self,
        state: "AgentState",
        success: bool,
    ) -> None:
        """End the root agent invocation span and detach its context."""
        span_and_token = self._invocation_spans.pop(state.run_id, None)
        if span_and_token is None:
            return

        span, token = span_and_token
        duration_ms = (state.updated_at - state.started_at).total_seconds() * 1000
        span.set_attributes(
            {
                "locus.success": success,
                "locus.iterations": state.iteration,
                "locus.confidence": state.confidence,
                "locus.tool_calls": len(state.tool_executions),
                "locus.errors": len(state.errors),
                "locus.duration_ms": duration_ms,
            }
        )
        if success:
            span.set_status(Status(StatusCode.OK))
        else:
            span.set_status(Status(StatusCode.ERROR, "Agent invocation failed"))

        span.end()
        otel_context.detach(token)

    async def on_iteration_start(
        self,
        iteration: int,
        state: "AgentState",
    ) -> None:
        """Start an iteration span under the active invocation span."""
        span = self._tracer.start_span(
            f"agent.iteration.{iteration}",
            attributes={
                "locus.iteration": iteration,
                "locus.confidence": state.confidence,
                "locus.messages": len(state.messages),
            },
        )
        token = otel_context.attach(trace.set_span_in_context(span))
        self._iteration_spans[(state.run_id, iteration)] = (span, token)

    async def on_iteration_end(
        self,
        iteration: int,
        state: "AgentState",
    ) -> None:
        """End an iteration span and restore the invocation context."""
        span_and_token = self._iteration_spans.pop((state.run_id, iteration), None)
        if span_and_token is None:
            return

        span, token = span_and_token
        span.set_attributes(
            {
                "locus.confidence_after": state.confidence,
                "locus.messages_after": len(state.messages),
            }
        )
        span.set_status(Status(StatusCode.OK))
        span.end()
        otel_context.detach(token)

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Start a tool span under the active iteration or invocation span."""
        span_attrs: dict[str, Any] = {"locus.tool_name": event.tool_name}

        span = self._tracer.start_span(f"tool.{event.tool_name}", attributes=span_attrs)
        token = otel_context.attach(trace.set_span_in_context(span))
        self._tool_spans[_tool_span_key(event)] = (span, token, time.perf_counter())

    async def on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        """End a tool span and restore the previous context."""
        span_and_token = self._tool_spans.pop(_tool_span_key(event), None)
        if span_and_token is None:
            return

        span, token, start_time = span_and_token
        duration_ms = (time.perf_counter() - start_time) * 1000
        span.set_attribute("locus.duration_ms", duration_ms)

        if event.error:
            span.set_status(Status(StatusCode.ERROR, event.error))
            span.set_attribute("locus.error", event.error[:1000])
        else:
            span.set_status(Status(StatusCode.OK))

        span.end()
        otel_context.detach(token)


def configure_langfuse_otel(config: DemoConfig) -> None:
    """Configure OpenTelemetry to export traces to Langfuse.

    Args:
        config: Demo configuration loaded from environment variables.

    Raises:
        ValueError: If Langfuse export is enabled without API keys.
    """
    global _LANGFUSE_OTEL_CONFIGURED  # pylint: disable=global-statement

    if _LANGFUSE_OTEL_CONFIGURED:
        return

    public_key = config.langfuse.public_key
    secret_key = config.langfuse.secret_key
    if not public_key or not secret_key:
        raise ValueError(
            "Langfuse telemetry is enabled, but "
            "DEMO_RAG_LANGFUSE_PUBLIC_KEY and "
            "DEMO_RAG_LANGFUSE_SECRET_KEY are required."
        )

    auth_string = base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode(
        "ascii"
    )
    exporter = OTLPSpanExporter(
        endpoint=config.langfuse.endpoint,
        headers={
            "Authorization": f"Basic {auth_string}",
            "x-langfuse-ingestion-version": "4",
        },
    )
    provider = TracerProvider(
        resource=Resource.create(
            {
                SERVICE_NAME: config.langfuse.service_name,
                "deployment.environment": config.langfuse.environment,
            }
        )
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _LANGFUSE_OTEL_CONFIGURED = True
    LOGGER.info(
        "Configured Langfuse OpenTelemetry export endpoint=%s service=%s",
        config.langfuse.endpoint,
        config.langfuse.service_name,
    )


def _tool_span_key(event: BeforeToolCallEvent | AfterToolCallEvent) -> str:
    """Return a stable key for matching before/after tool hook events."""
    tool_call_id = getattr(event, "tool_call_id", "")
    task = asyncio.current_task()
    task_id = id(task) if task is not None else 0
    return tool_call_id or f"{task_id}:{event.tool_name}"
