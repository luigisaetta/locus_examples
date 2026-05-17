"""Query the RAG AgentServer from the command line."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import httpx

from demos.demo_rag.common import configure_logging
from demos.demo_rag.config import DemoConfig, load_config

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Send a query to the demo RAG AgentServer.",
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="Question to send to the running AgentServer.",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Optional AgentServer thread id.",
    )
    return parser.parse_args()


def build_invoke_url(config: DemoConfig) -> str:
    """Build the AgentServer invoke endpoint URL.

    Args:
        config: Demo configuration loaded from environment variables.

    Returns:
        Fully qualified `/invoke` endpoint URL.
    """
    base_url = config.agent_server.url.rstrip("/")
    return f"{base_url}/invoke"


def build_payload(query: str, thread_id: str | None = None) -> dict[str, Any]:
    """Build the AgentServer invoke request body.

    Args:
        query: User query to send to the agent.
        thread_id: Optional AgentServer thread id.

    Returns:
        JSON-serializable request payload.
    """
    payload: dict[str, Any] = {
        "prompt": query,
        "metadata": {
            "client": "demos.demo_rag.query_agent",
        },
    }
    if thread_id:
        payload["thread_id"] = thread_id
    return payload


def invoke_agent(config: DemoConfig, query: str, thread_id: str | None = None) -> Any:
    """Invoke the RAG AgentServer and return the decoded response.

    Args:
        config: Demo configuration loaded from environment variables.
        query: User query to send to the agent.
        thread_id: Optional AgentServer thread id.

    Returns:
        Decoded JSON response from the server.

    Raises:
        httpx.HTTPError: If the HTTP call fails or returns an error status.
    """
    url = build_invoke_url(config)
    payload = build_payload(query, thread_id)

    LOGGER.info("POST %s", url)
    with httpx.Client(timeout=120.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


def print_result(result: Any) -> None:
    """Print a complete AgentServer result as formatted JSON.

    Args:
        result: Decoded response object returned by the AgentServer.
    """
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    """Run the command-line AgentServer client."""
    config = load_config()
    configure_logging(config.runtime.log_level)
    args = parse_args()
    query = " ".join(args.query)

    result = invoke_agent(config, query, thread_id=args.thread_id)
    print_result(result)


if __name__ == "__main__":
    main()
