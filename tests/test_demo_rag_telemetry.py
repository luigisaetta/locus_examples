"""
Author: L. Saetta
Last update: 2026-05-20
License: MIT
Description: Tests for RAG demo Langfuse telemetry hook configuration.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from demos.demo_rag import telemetry


class FakeTelemetryHook:  # pylint: disable=too-few-public-methods
    """Capture telemetry hook constructor arguments."""

    def __init__(self, **kwargs: Any) -> None:
        """Store telemetry hook settings."""
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        """Return the same name as the official telemetry hook."""
        return "TelemetryHook"


def _config(enabled: bool = True, public_key: str | None = "pk-test") -> Any:
    """Build the minimal demo config needed by telemetry helpers."""
    return SimpleNamespace(
        langfuse=SimpleNamespace(
            enabled=enabled,
            endpoint="https://cloud.langfuse.com/api/public/otel/v1/traces",
            public_key=public_key,
            secret_key="sk-test",
            service_name="locus-demo-rag-a2a",
            environment="test",
        )
    )


def test_build_hooks_returns_no_hooks_when_langfuse_disabled() -> None:
    """Verify disabled Langfuse telemetry does not install hooks."""
    assert not telemetry.build_hooks(_config(enabled=False))


def test_build_hooks_uses_locus_telemetry_hook(monkeypatch: Any) -> None:
    """Verify enabled Langfuse telemetry installs the official Locus hook."""
    monkeypatch.setattr(telemetry, "configure_langfuse_otel", lambda config: None)
    monkeypatch.setattr(telemetry, "TelemetryHook", FakeTelemetryHook)

    hooks = telemetry.build_hooks(_config())

    assert len(hooks) == 1
    assert hooks[0].name == "TelemetryHook"
    assert hooks[0].kwargs == {
        "service_name": "locus-demo-rag-a2a",
        "record_arguments": False,
        "record_results": False,
    }


def test_configure_langfuse_otel_requires_keys(monkeypatch: Any) -> None:
    """Verify Langfuse telemetry fails clearly when credentials are missing."""
    monkeypatch.setattr(telemetry, "_LANGFUSE_OTEL_CONFIGURED", False)

    with pytest.raises(ValueError, match="DEMO_RAG_LANGFUSE_PUBLIC_KEY"):
        telemetry.configure_langfuse_otel(_config(public_key=None))
