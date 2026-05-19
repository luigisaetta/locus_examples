"""
Author: L. Saetta
Last update: 2026-05-19
License: MIT
Description: Command-line client for querying the RAG A2A server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from typing import Any

from locus.a2a import A2AClient, Message, TextPart

from demos.demo_rag.common import configure_logging
from demos.demo_rag.config import DemoConfig, load_config

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fetch the A2A Agent Card and send a query to the RAG A2A server.",
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="Question to send to the running A2A server.",
    )
    return parser.parse_args()


def build_client(config: DemoConfig) -> A2AClient:
    """Create an A2A client from demo configuration.

    Args:
        config: Demo configuration loaded from environment variables.

    Returns:
        Configured A2A client.
    """
    return A2AClient(
        url=config.a2a_server.url,
        api_key=config.a2a_server.api_key,
    )


def build_message(query: str) -> Message:
    """Build an A2A user message for `message/send`.

    Args:
        query: User query to send to the remote A2A agent.

    Returns:
        A2A Message containing the query as a text part.
    """
    return Message(
        role="user",
        parts=[TextPart(text=query)],
        messageId=uuid.uuid4().hex,
        metadata={
            "client": "demos.demo_rag.query_a2a",
        },
    )


def _to_plain_dict(value: Any) -> dict[str, Any]:
    """Convert an A2A model or mapping to a plain dictionary.

    Args:
        value: JSON-serializable value or Pydantic model.

    Returns:
        Plain dictionary representation.
    """
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    return dict(value)


def print_json(title: str, value: Any) -> None:
    """Print a titled JSON value.

    Args:
        title: Human-readable section title.
        value: JSON-serializable value or Pydantic model.
    """
    print(f"\n=== {title} ===")
    print(json.dumps(_to_plain_dict(value), indent=2, ensure_ascii=False, default=str))


def _extract_final_answer(task: dict[str, Any]) -> str:
    """Extract the final agent answer from an A2A task.

    Args:
        task: Plain A2A task dictionary.

    Returns:
        Final text answer, or an empty string when absent.
    """
    artifacts = task.get("artifacts") or []
    for artifact in reversed(artifacts):
        for part in artifact.get("parts", []):
            text = part.get("text")
            if text:
                return str(text)
    status = task.get("status") or {}
    message = status.get("message") or {}
    for part in message.get("parts", []):
        text = part.get("text")
        if text:
            return str(text)
    return ""


def print_result(task: Any) -> None:
    """Print the final answer and A2A task metadata.

    Args:
        task: A2A task returned by the server.
    """
    task_dict = _to_plain_dict(task)
    metadata = {
        "id": task_dict.get("id"),
        "contextId": task_dict.get("contextId"),
        "status": task_dict.get("status"),
        "metadata": task_dict.get("metadata"),
    }
    output = {
        "answer": _extract_final_answer(task_dict),
        "metadata": metadata,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False, default=str))


async def run_query(config: DemoConfig, query: str) -> None:
    """Fetch the Agent Card, invoke the A2A agent, and print both results.

    Args:
        config: Demo configuration loaded from environment variables.
        query: User query to send to the remote A2A agent.
    """
    client = build_client(config)

    LOGGER.info("Fetching A2A Agent Card from %s", config.a2a_server.url)
    card = await client.get_agent_card()
    print_json("A2A Agent Card", card)

    LOGGER.info("Sending A2A message/send request")
    task = await client.send_message(build_message(query))
    print_result(task)


def main() -> None:
    """Run the command-line A2A client."""
    config = load_config()
    configure_logging(config.runtime.log_level)
    args = parse_args()
    query = " ".join(args.query)

    asyncio.run(run_query(config, query))


if __name__ == "__main__":
    main()
