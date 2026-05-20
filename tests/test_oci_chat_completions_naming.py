"""
Author: L. Saetta
Last update: 2026-05-20
License: MIT
Description: Tests for OCI chat completions model naming in examples.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from locus.models.providers import oci as oci_provider

from demos.demo_rag import common as rag_common
from examples import config as examples_config


# pylint: disable=too-few-public-methods
class FakeOCIChatCompletionsModel:
    """Capture OCI chat completions constructor arguments."""

    def __init__(self, **kwargs: Any) -> None:
        """Store the constructor keyword arguments for assertions."""
        self.kwargs = kwargs


def test_locus_provider_exports_chat_completions_names() -> None:
    """Verify the renamed OCI chat completions classes are importable."""
    assert oci_provider.OCIChatCompletionsModel.__name__ == "OCIChatCompletionsModel"
    assert oci_provider.OCIChatCompletionsConfig.__name__ == "OCIChatCompletionsConfig"


def test_examples_config_uses_chat_completions_model(monkeypatch: Any) -> None:
    """Verify example OCI v1 transport builds the renamed model class."""
    monkeypatch.setattr(
        oci_provider,
        "OCIChatCompletionsModel",
        FakeOCIChatCompletionsModel,
    )
    monkeypatch.setenv("LOCUS_MODEL_PROVIDER", "oci")
    monkeypatch.setenv("LOCUS_MODEL_ID", "openai.test-model")
    monkeypatch.setenv("LOCUS_OCI_PROFILE", "TEST_PROFILE")
    monkeypatch.setenv("LOCUS_OCI_REGION", "eu-frankfurt-1")
    monkeypatch.setenv("LOCUS_OCI_TRANSPORT", "v1")
    monkeypatch.delenv("LOCUS_OCI_AUTH_TYPE", raising=False)
    monkeypatch.delenv("LOCUS_OCI_COMPARTMENT", raising=False)

    model = examples_config.get_model()

    assert isinstance(model, FakeOCIChatCompletionsModel)
    assert model.kwargs == {
        "model": "openai.test-model",
        "profile": "TEST_PROFILE",
        "compartment_id": None,
        "region": "eu-frankfurt-1",
    }


def test_rag_agent_model_uses_chat_completions_model(monkeypatch: Any) -> None:
    """Verify the RAG demo OCI model builder uses the renamed model class."""
    monkeypatch.setattr(
        rag_common,
        "OCIChatCompletionsModel",
        FakeOCIChatCompletionsModel,
    )
    config = SimpleNamespace(
        agent_model="oci:openai.test-model",
        embeddings=SimpleNamespace(
            auth_type="api_key",
            profile_name="TEST_PROFILE",
            compartment_id="ocid1.compartment.oc1..test",
            region="eu-frankfurt-1",
            config_file="/tmp/oci-config",
        ),
    )

    model = rag_common.build_agent_model(config)

    assert isinstance(model, FakeOCIChatCompletionsModel)
    assert model.kwargs == {
        "model": "openai.test-model",
        "profile": "TEST_PROFILE",
        "compartment_id": "ocid1.compartment.oc1..test",
        "region": "eu-frankfurt-1",
        "base_url": "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com/openai/v1",
        "config_file": "/tmp/oci-config",
    }
