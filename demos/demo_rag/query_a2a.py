"""Query the RAG A2A server from the command line."""

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


def print_json(title: str, value: Any) -> None:
    """Print a titled JSON value.

    Args:
        title: Human-readable section title.
        value: JSON-serializable value or Pydantic model.
    """
    if hasattr(value, "model_dump"):
        value = value.model_dump()

    print(f"\n=== {title} ===")
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


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
    print_json("A2A Task Result", task)


def main() -> None:
    """Run the command-line A2A client."""
    config = load_config()
    configure_logging(config.runtime.log_level)
    args = parse_args()
    query = " ".join(args.query)

    asyncio.run(run_query(config, query))


if __name__ == "__main__":
    main()
