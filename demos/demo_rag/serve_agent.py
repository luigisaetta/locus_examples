"""Serve a RAG-enabled Locus Agent backed by OracleVectorStore."""

from __future__ import annotations

import logging

from locus.agent import Agent
from locus.rag import create_rag_tool
from locus.server import AgentServer

from demos.demo_rag.common import configure_logging, configure_warnings, build_retriever
from demos.demo_rag.config import DemoConfig, load_config

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a helpful assistant.
Use the search_knowledge tool when the user asks about information that may be
stored in the document knowledge base.
"""


def build_agent(config: DemoConfig) -> Agent:
    """Create a Locus Agent with a RAG search tool.

    Args:
        config: Demo configuration loaded from environment variables.

    Returns:
        Configured Locus Agent using OracleVectorStore through RAGRetriever.
    """
    LOGGER.info("Creating RAG retriever for agent tool")
    retriever = build_retriever(config)

    LOGGER.info("Creating Agent with model: %s", config.agent_model)
    return Agent(
        model=config.agent_model,
        tools=[create_rag_tool(retriever)],
        system_prompt=SYSTEM_PROMPT,
    )


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

    LOGGER.info("Starting RAG AgentServer")
    server = build_server(config)

    server.run(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
