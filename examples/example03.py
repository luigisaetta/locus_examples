"""Example03: demonstrate a simple Locus StateGraph with async functions."""

import time
import asyncio

from config import get_model
from locus.agent import Agent
from locus.multiagent import END, START, StateGraph


def _llm_call(
    prompt: str, *, system: str = "Reply in one short sentence.", max_tokens: int = 2000
) -> str:
    """Helper: real model call with timing/token banner — used by every Part."""
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


async def example_state_flow():
    """See how state accumulates through nodes."""
    print("=== State Flow ===\n")

    graph = StateGraph()

    async def step_a(inputs):
        print(f"  Step A receives: {list(inputs.keys())}")

        # as in LangGraph, every node returns a dict that gets merged into the state.
        return {"a_output": "from A", "value": 10}

    async def step_b(inputs):
        print(f"  Step B receives: {list(inputs.keys())}")
        value = inputs.get("value", 0)
        return {"b_output": "from B", "doubled": value * 2}

    async def step_c(inputs):
        print(f"  Step C receives: {list(inputs.keys())}")
        # Final node delegates to a real Agent.
        doubled = inputs.get("doubled", 0)

        ai = _llm_call(
            f"Briefly comment on a graph that doubled the value to {doubled}.",
        )
        return {"c_output": "from C", "final": doubled + 5, "ai_comment": ai}

    # build the graph
    graph.add_node("step_a", step_a)
    graph.add_node("step_b", step_b)
    graph.add_node("step_c", step_c)

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
    """Run the state-flow example from the command line."""
    await example_state_flow()


if __name__ == "__main__":
    asyncio.run(main())
