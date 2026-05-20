"""
Author: L. Saetta
Last update: 2026-05-20
License: MIT
Description: Tests for examples that must run as modules from the repository root.
"""

from __future__ import annotations

import importlib


def test_state_graph_examples_import_as_modules() -> None:
    """Verify examples using shared config import cleanly as modules."""
    for module_name in (
        "examples.example03",
        "examples.example04",
        "examples.example05",
    ):
        module = importlib.import_module(module_name)

        assert module is not None
