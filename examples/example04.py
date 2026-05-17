"""Example04: build a Locus ``StateGraph`` with callable step objects.

This example is the class-based counterpart of ``example03``.  Instead of
registering plain async functions as graph nodes, each node is an instance of a
class that implements ``async __call__``.  Locus accepts these objects because
``StateGraph.add_node`` expects a ``Callable`` executor.

The graph demonstrates the same state-merge behavior as the function-based
example:

1. ``StepA`` adds a base value to the shared graph state.
2. ``StepB`` reads that value and adds a derived value.
3. ``StepC`` calls a real LLM through ``Agent`` and adds the final output.
"""

# pylint: disable=duplicate-code,too-few-public-methods

import asyncio
import time
from typing import Any

from config import get_model
from locus.agent import Agent
from locus.multiagent import END, START, StateGraph


def _llm_call(
    prompt: str, *, system: str = "Reply in one short sentence.", max_tokens: int = 2000
) -> str:
    """Run a synchronous Locus ``Agent`` call and return the model text.

    Args:
        prompt: User prompt sent to the model.
        system: System instruction used to configure the temporary agent.
        max_tokens: Maximum number of output tokens requested from the model.

    Returns:
        The stripped assistant message produced by the model.

    Side effects:
        Prints a small timing and token-usage banner.  This keeps the example
        transparent when it performs a real network-backed model call.
    """
    agent = Agent(
        model=get_model(max_tokens=max_tokens, temperature=0.0), system_prompt=system
    )

    t0 = time.perf_counter()
    res = agent.run_sync(prompt)
    dt = time.perf_counter() - t0

    banner = (
        f"  [model call: {dt:.2f}s · "
        f"{res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    print(banner)
    return res.message.strip()


class StepA:
    """First graph node implemented as an async callable object.

    Locus invokes the instance as ``await step(inputs)``.  The returned mapping
    is merged into the graph state and becomes visible to downstream nodes.
    """

    async def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Seed the graph state with a numeric value.

        Args:
            inputs: Current graph state received from upstream nodes.

        Returns:
            A state update containing a trace marker and the initial numeric
            value used by ``StepB``.
        """
        print(f"  Step A receives: {list(inputs.keys())}")
        return {"a_output": "from A", "value": 10}


class StepB:
    """Second graph node that consumes state produced by ``StepA``."""

    async def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Double the ``value`` entry from the accumulated graph state.

        Args:
            inputs: Current graph state.  ``value`` is expected if ``StepA`` ran,
                but defaults to ``0`` to keep the node defensive.

        Returns:
            A state update with a trace marker and the doubled value.
        """
        print(f"  Step B receives: {list(inputs.keys())}")
        value = inputs.get("value", 0)
        return {"b_output": "from B", "doubled": value * 2}


class StepC:
    """Final graph node that combines deterministic state with an LLM call."""

    async def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Produce the final numeric result and an AI-generated comment.

        Args:
            inputs: Current graph state, including ``doubled`` from ``StepB``.

        Returns:
            A state update containing:
            - ``c_output``: a trace marker for the node.
            - ``final``: the deterministic final value.
            - ``ai_comment``: a short model-generated comment.
        """
        print(f"  Step C receives: {list(inputs.keys())}")
        doubled = inputs.get("doubled", 0)

        ai = _llm_call(
            f"Briefly comment on a graph that doubled the value to {doubled}.",
        )
        return {"c_output": "from C", "final": doubled + 5, "ai_comment": ai}


async def example_state_flow():
    """Create and execute a three-node stateful graph.

    The graph starts with ``{"initial_data": True}``.  Each node receives the
    accumulated state, returns a dictionary, and Locus merges that dictionary
    into the state passed to the next node.

    The shape is:

    ``START -> step_a -> step_b -> step_c -> END``
    """
    print("=== State Flow ===\n")

    graph = StateGraph()

    # build the graph
    graph.add_node("step_a", StepA())
    graph.add_node("step_b", StepB())
    graph.add_node("step_c", StepC())

    graph.add_edge(START, "step_a")
    graph.add_edge("step_a", "step_b")
    graph.add_edge("step_b", "step_c")
    graph.add_edge("step_c", END)

    print("Executing graph...")
    result = await graph.execute({"initial_data": True})

    print("\nFinal state:")
    for key, value in result.final_state.items():
        if not key.startswith("_"):  # Skip internal keys
            print(f"  {key}: {value}")
    print()


async def main():
    """Program entry point used when running the example as a script."""
    await example_state_flow()


if __name__ == "__main__":
    asyncio.run(main())
