"""Serve a RAG-enabled Locus Agent backed by OracleVectorStore."""

from __future__ import annotations

import logging

from locus.server import AgentServer

from demos.demo_rag.common import (
    build_agent,
    configure_logging,
    configure_warnings,
)
from demos.demo_rag.config import DemoConfig, load_config

LOGGER = logging.getLogger(__name__)


def build_server(config: DemoConfig) -> AgentServer:
    """Create an AgentServer wrapping the RAG-enabled agent.

    Args:
        config: Demo configuration loaded from environment variables.

    Returns:
        AgentServer exposing invoke, stream, thread, and health endpoints.
    """
    agent = build_agent(config)
    return AgentServer(
        agent=agent,
        title="Locus RAG Demo Agent",
        description="A Locus Agent with OracleVectorStore-backed RAG search.",
        allow_unauthenticated=True,
    )


def main() -> None:
    """Run the RAG AgentServer from the command line."""
    configure_warnings()
    config = load_config()
    configure_logging(config.runtime.log_level)

    LOGGER.info(
        "Starting RAG AgentServer at %s:%s",
        config.agent_server.host,
        config.agent_server.port,
    )
    server = build_server(config)

    server.run(host=config.agent_server.host, port=config.agent_server.port)


if __name__ == "__main__":
    main()
