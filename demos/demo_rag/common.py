"""Shared builders for the RAG demo components."""

from __future__ import annotations

import logging
import warnings

from locus.rag.embeddings.oci import OCIEmbeddings
from locus.rag.retriever import RAGRetriever
from locus.rag.stores.oracle import OracleVectorStore

from demos.demo_rag.config import DemoConfig

LOGGER = logging.getLogger(__name__)


def configure_warnings() -> None:
    """Hide known third-party warnings that are not actionable for this demo."""
    warnings.filterwarnings(
        "ignore",
        message="The 'strict' parameter is no longer needed on Python 3\\+.*",
        category=FutureWarning,
        module="urllib3.poolmanager",
    )


def configure_logging(level: str) -> None:
    """Configure console logging for demo scripts.

    Args:
        level: Logging level name, for example INFO or DEBUG.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def build_embedder(config: DemoConfig) -> OCIEmbeddings:
    """Create the Locus OCI embedding provider from demo configuration.

    Args:
        config: Demo configuration loaded from environment variables.

    Returns:
        Configured Locus OCI embeddings provider.
    """
    LOGGER.info("Creating OCIEmbeddings with model: %s", config.embeddings.model_id)
    return OCIEmbeddings(
        model_id=config.embeddings.model_id,
        compartment_id=config.embeddings.compartment_id,
        profile_name=config.embeddings.profile_name,
        auth_type=config.embeddings.auth_type,
        service_endpoint=config.embeddings.service_endpoint,
        config_file=config.embeddings.config_file,
    )


def build_store(config: DemoConfig, dimension: int) -> OracleVectorStore:
    """Create the Locus OracleVectorStore from demo configuration.

    Args:
        config: Demo configuration loaded from environment variables.
        dimension: Embedding vector dimension to use for the Oracle table.

    Returns:
        Configured Oracle vector store.
    """
    LOGGER.info(
        "Creating OracleVectorStore table=%s schema=%s dimension=%s metric=%s",
        config.oracle.table_name,
        config.oracle.schema_name or "<default user>",
        dimension,
        config.oracle.distance_metric,
    )
    return OracleVectorStore(
        dsn=config.oracle.dsn,
        user=config.oracle.user,
        password=config.oracle.password,
        wallet_location=config.oracle.wallet.location,
        wallet_password=config.oracle.wallet.password,
        table_name=config.oracle.table_name,
        schema_name=config.oracle.schema_name,
        dimension=dimension,
        distance_metric=config.oracle.distance_metric,
    )


def build_retriever(config: DemoConfig) -> RAGRetriever:
    """Create the Locus RAGRetriever used by demo scripts and agents.

    Args:
        config: Demo configuration loaded from environment variables.

    Returns:
        Configured Locus RAG retriever backed by OracleVectorStore.
    """
    embedder = build_embedder(config)
    store = build_store(config, dimension=embedder.config.dimension)
    LOGGER.info(
        "Creating RAGRetriever chunk_size=%s chunk_overlap=%s",
        config.runtime.chunk_size,
        config.runtime.chunk_overlap,
    )
    return RAGRetriever(
        embedder=embedder,
        store=store,
        chunk_size=config.runtime.chunk_size,
        chunk_overlap=config.runtime.chunk_overlap,
    )
