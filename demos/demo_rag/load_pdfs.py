"""Load root-level PDF documents into an OracleVectorStore using Locus."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from pathlib import Path

from locus.rag.retriever import RAGRetriever

from demos.demo_rag.common import configure_logging, configure_warnings, build_retriever
from demos.demo_rag.config import DemoConfig, load_config

LOGGER = logging.getLogger(__name__)


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
        if config.runtime.clear_before_load:
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
    configure_warnings()
    config = load_config()
    configure_logging(config.runtime.log_level)

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
