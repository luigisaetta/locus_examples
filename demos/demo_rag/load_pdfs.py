"""Load root-level PDF documents into an OracleVectorStore using Locus."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from pathlib import Path

from locus.rag.embeddings.oci import OCIEmbeddings
from locus.rag.retriever import RAGRetriever
from locus.rag.stores.oracle import OracleVectorStore

from demos.demo_rag.config import DemoConfig, load_config

LOGGER = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    """Configure console logging for the demo script.

    Args:
        level: Logging level name, for example INFO or DEBUG.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def find_pdf_files(pdf_dir: Path) -> list[Path]:
    """Return all PDF files contained directly or recursively in a folder.

    Args:
        pdf_dir: Folder to scan for PDF files.

    Returns:
        Sorted list of discovered PDF file paths.

    Raises:
        FileNotFoundError: If the configured PDF folder does not exist.
        NotADirectoryError: If the configured PDF path is not a folder.
    """
    LOGGER.info("Checking PDF folder: %s", pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF folder not found: {pdf_dir}")
    if not pdf_dir.is_dir():
        raise NotADirectoryError(f"PDF path is not a folder: {pdf_dir}")

    pdf_files = sorted(path for path in pdf_dir.rglob("*.pdf") if path.is_file())
    LOGGER.info("Found %s PDF file(s)", len(pdf_files))
    return pdf_files


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
    """Create the Locus RAGRetriever used by the loader.

    Args:
        config: Demo configuration loaded from environment variables.

    Returns:
        Configured Locus RAG retriever backed by OracleVectorStore.
    """
    embedder = build_embedder(config)
    store = build_store(config, dimension=embedder.config.dimension)
    LOGGER.info(
        "Creating RAGRetriever chunk_size=%s chunk_overlap=%s",
        config.chunk_size,
        config.chunk_overlap,
    )
    return RAGRetriever(
        embedder=embedder,
        store=store,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )


async def load_pdf(
    retriever: RAGRetriever, pdf_path: Path, source_root: Path
) -> list[str]:
    """Load one PDF file into the configured vector store.

    Args:
        retriever: Locus RAG retriever configured with OracleVectorStore.
        pdf_path: PDF file path to load.
        source_root: Root folder used to compute source metadata.

    Returns:
        Document chunk IDs created in the vector store.
    """
    relative_path = pdf_path.relative_to(source_root)
    LOGGER.info("Loading PDF: %s", relative_path)

    document_ids = await retriever.add_file(
        pdf_path,
        metadata={
            "source": str(relative_path),
            "source_type": "pdf",
        },
        chunk=True,
    )

    LOGGER.info("Loaded %s chunk(s) from %s", len(document_ids), relative_path)
    return document_ids


async def load_all_pdfs(config: DemoConfig, pdf_files: Iterable[Path]) -> int:
    """Load every discovered PDF file and return the number of stored chunks.

    Args:
        config: Demo configuration loaded from environment variables.
        pdf_files: PDF file paths to load.

    Returns:
        Total number of document chunks stored in OracleVectorStore.
    """
    retriever = build_retriever(config)
    total_chunks = 0

    try:
        if config.clear_before_load:
            LOGGER.info("Clearing vector store before loading new documents")
            removed_count = await retriever.store.clear()
            LOGGER.info("Cleared %s existing document chunk(s)", removed_count)

        for pdf_path in pdf_files:
            document_ids = await load_pdf(retriever, pdf_path, config.pdf_dir)
            total_chunks += len(document_ids)
    finally:
        LOGGER.info("Closing OracleVectorStore resources")
        await retriever.store.close()

    return total_chunks


async def run() -> None:
    """Run the PDF loading workflow."""
    config = load_config()
    configure_logging(config.log_level)

    LOGGER.info("Starting OracleVectorStore PDF loader demo")
    LOGGER.info("Using table name: %s", config.oracle.table_name)

    pdf_files = find_pdf_files(config.pdf_dir)
    if not pdf_files:
        LOGGER.warning("No PDF files found. Nothing to load.")
        return

    total_chunks = await load_all_pdfs(config, pdf_files)
    LOGGER.info("Finished loading %s PDF file(s)", len(pdf_files))
    LOGGER.info("Stored %s document chunk(s)", total_chunks)


def main() -> None:
    """Run the async demo entry point from the command line."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
