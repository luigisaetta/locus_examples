"""
Author: L. Saetta
Last update: 2026-05-19
License: MIT
Description: Example showing how to expose a StateGraph through AgentServer.

``AgentServer`` is designed around the Locus Agent runtime shape: it calls an
object's async ``run(prompt, thread_id=..., metadata=...)`` method and consumes
Locus events such as ``ThinkEvent`` and ``TerminateEvent``.

``StateGraph`` has a different native API: ``execute(inputs)`` and
``stream(inputs)`` return graph-specific result/event objects.  This example
shows the small adapter layer needed to make a graph look like an agent to the
server, without changing the graph itself.

Run:

    python examples/example05.py

Then invoke:

    curl -X POST http://127.0.0.1:8000/invoke \
      -H "Content-Type: application/json" \
      -d '{"prompt":"Explain why callable graph nodes are useful."}'
"""

# pylint: disable=too-few-public-methods

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from config import get_model
from locus.agent import Agent
from locus.core.events import LocusEvent, TerminateEvent, ThinkEvent
from locus.multiagent import END, START, StateGraph
from locus.multiagent.graph import StreamMode
from locus.server import AgentServer


def _llm_call(prompt: str) -> str:
    """Call a real or mock model and return a short response.

    The example uses the shared ``get_model`` helper, so it works with the mock
    provider by default and can be pointed at OCI/OpenAI by environment
    variables.
    """
    agent = Agent(
        model=get_model(max_tokens=400, temperature=0.0),
        system_prompt="Reply in one concise paragraph.",
    )
    return agent.run_sync(prompt).message.strip()


async def prepare_prompt(inputs: dict[str, Any]) -> dict[str, Any]:
    """Normalize the HTTP prompt into the first graph state update."""
    prompt = inputs.get("prompt", "")
    return {
        "original_prompt": prompt,
        "prepared_prompt": (
            "Answer this request as a graph-backed service. " f"User request: {prompt}"
        ),
    }


async def answer_with_model(inputs: dict[str, Any]) -> dict[str, Any]:
    """Use a Locus ``Agent`` inside a graph node to produce the answer."""
    answer = _llm_call(inputs.get("prepared_prompt", ""))
    return {"answer": answer}


async def format_response(inputs: dict[str, Any]) -> dict[str, Any]:
    """Select the graph field that should become the HTTP response body."""
    return {
        "final_message": inputs.get("answer", "The graph completed without an answer.")
    }


def build_graph() -> StateGraph:
    """Build the graph served by ``AgentServer``.

    The graph is intentionally small:

    ``START -> prepare -> answer -> format -> END``

    Each node returns a dictionary that Locus merges into the shared graph
    state.  ``GraphAgent`` later reads ``final_message`` from that final state.
    """
    graph = StateGraph()

    graph.add_node("prepare", prepare_prompt)
    graph.add_node("answer", answer_with_model)
    graph.add_node("format", format_response)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "answer")
    graph.add_edge("answer", "format")
    graph.add_edge("format", END)

    return graph


class GraphAgent:
    """Adapter that makes a ``StateGraph`` compatible with ``AgentServer``.

    ``AgentServer`` does not require a concrete ``Agent`` instance, but it does
    expect the object to expose:

    - ``run(prompt, thread_id=None, metadata=None)``, yielding Locus events.
    - ``config.checkpointer`` if the thread endpoints are used.

    The adapter keeps those requirements small and explicit.
    """

    def __init__(self, graph: StateGraph, output_key: str = "final_message") -> None:
        """Store the wrapped graph and the final-state key to expose."""
        self.graph = graph
        self.config = graph.config
        self.output_key = output_key

    async def run(
        self,
        prompt: str,
        *,
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[LocusEvent]:
        """Run the graph and translate graph events into AgentServer events.

        Args:
            prompt: HTTP prompt received by ``AgentServer``.
            thread_id: Optional server-scoped thread id.  It is copied into the
                graph config so graph checkpointing can use it.
            metadata: Optional request metadata.  It is placed in the initial
                graph state for nodes that need it.

        Yields:
            ``ThinkEvent`` for node progress and one final ``TerminateEvent``
            containing the response message.
        """
        config = self.graph.config.model_copy(
            update={
                "thread_id": thread_id,
                "stream_mode": StreamMode.VALUES,
            }
        )
        inputs = {
            "prompt": prompt,
            "metadata": metadata or {},
        }
        final_state: dict[str, Any] | None = None
        node_events = 0

        async for event in self.graph.stream(
            inputs, config=config, mode=StreamMode.VALUES
        ):
            if event.node_id is None:
                final_state = dict(event.data or {})
                continue

            node_id = event.node_id or "graph"
            node_events += 1
            yield ThinkEvent(
                iteration=node_events,
                reasoning=f"Graph node completed: {node_id}",
            )

        if final_state is None:
            final_state = {}

        final_message = self._extract_final_message(final_state)

        yield TerminateEvent(
            reason="complete",
            iterations_used=node_events,
            final_confidence=1.0,
            total_tool_calls=0,
            final_message=final_message,
        )

    def _extract_final_message(self, final_state: dict[str, Any]) -> str:
        """Return a human-facing response from a graph final state."""
        candidate = (
            final_state.get(self.output_key)
            or final_state.get("answer")
            or final_state.get("message")
        )
        if candidate is not None:
            return str(candidate)
        return str(final_state)


def main() -> None:
    """Start a local ``AgentServer`` backed by a ``StateGraph``."""
    graph = build_graph()
    graph_agent = GraphAgent(graph)

    server = AgentServer(
        agent=graph_agent,
        title="Locus Graph Agent Server",
        description="A StateGraph exposed through the AgentServer adapter shape.",
        allow_unauthenticated=True,
    )
    server.run(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
