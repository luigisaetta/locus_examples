"""
Author: L. Saetta
Last update: 2026-05-20
License: MIT
Description: OpenTelemetry and Langfuse hook helpers for the RAG demo.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from locus.hooks.builtin.telemetry import TelemetryHook
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from demos.demo_rag.config import DemoConfig

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
    return [
        TelemetryHook(
            service_name=config.langfuse.service_name,
            record_arguments=False,
            record_results=False,
        )
    ]


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
