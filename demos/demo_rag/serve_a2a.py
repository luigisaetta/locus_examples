"""Serve the RAG-enabled Locus Agent through the A2A protocol."""

from __future__ import annotations

import logging

from locus.a2a import A2AServer, AgentSkill

from demos.demo_rag.common import configure_logging, configure_warnings
from demos.demo_rag.config import DemoConfig, load_config
from demos.demo_rag.serve_agent import build_agent

LOGGER = logging.getLogger(__name__)


def build_a2a_server(config: DemoConfig) -> A2AServer:
    """Create an A2A server for the RAG search agent.

    Args:
        config: Demo configuration loaded from environment variables.

    Returns:
        A configured A2AServer wrapping the RAG-enabled Locus Agent.
    """
    agent = build_agent(config)
    return A2AServer(
        agent=agent,
        name="locus_rag_search",
        description=(
            "Answers questions using only documents retrieved from the "
            "OracleVectorStore-backed knowledge base."
        ),
        url=config.a2a_server.url,
        skills=[
            AgentSkill(
                id="knowledge_search",
                name="Knowledge search",
                description="Search loaded PDF documents and answer from retrieved chunks.",
                tags=["rag", "pdf", "oracle-vector-store"],
            ),
            AgentSkill(
                id="document_grounded_answer",
                name="Document-grounded answer",
                description=(
                    "Answer questions using only evidence returned by the "
                    "knowledge search tool."
                ),
                tags=["grounded-answer", "retrieval"],
            ),
        ],
        api_key=config.a2a_server.api_key,
    )


def main() -> None:
    """Run the A2A server from the command line."""
    configure_warnings()
    config = load_config()
    configure_logging(config.runtime.log_level)

    LOGGER.info(
        "Starting RAG A2A server at %s:%s",
        config.a2a_server.host,
        config.a2a_server.port,
    )
    server = build_a2a_server(config)
    server.run(host=config.a2a_server.host, port=config.a2a_server.port)


if __name__ == "__main__":
    main()
