"""Configuration helpers for the OracleVectorStore RAG demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
DEFAULT_ENV_FILE = DEMO_DIR / ".env"


@dataclass(frozen=True)
class OracleWalletConfig:
    """Oracle wallet configuration.

    Attributes:
        location: Optional wallet directory for Autonomous Database.
        password: Optional wallet password.
    """

    location: str | None
    password: str | None


@dataclass(frozen=True)
class OracleStoreConfig:
    """OracleVectorStore configuration.

    Attributes:
        dsn: Oracle DSN or TNS alias used by OracleVectorStore.
        user: Oracle database user.
        password: Oracle database password.
        wallet: Optional Oracle wallet configuration.
        table_name: Vector table name used by OracleVectorStore.
        schema_name: Optional schema name for the vector table.
        distance_metric: Vector distance metric, for example COSINE.
    """

    dsn: str | None
    user: str
    password: str
    wallet: OracleWalletConfig
    table_name: str
    schema_name: str | None
    distance_metric: str


@dataclass(frozen=True)
class OCIEmbeddingDemoConfig:
    """OCI embedding configuration.

    Attributes:
        model_id: OCI GenAI embedding model ID.
        compartment_id: OCI compartment OCID for GenAI calls.
        profile_name: OCI config profile name.
        auth_type: OCI authentication type.
        service_endpoint: Optional OCI GenAI service endpoint.
        config_file: OCI config file path.
    """

    model_id: str
    compartment_id: str
    profile_name: str
    auth_type: str
    service_endpoint: str | None
    config_file: str


@dataclass(frozen=True)
class RuntimeConfig:
    """Demo runtime behavior configuration.

    Attributes:
        chunk_size: Maximum chunk size used by RAGRetriever.
        chunk_overlap: Chunk overlap used by RAGRetriever.
        clear_before_load: Whether to clear the vector table before loading.
        log_level: Console logging level.
    """

    chunk_size: int
    chunk_overlap: int
    clear_before_load: bool
    log_level: str


@dataclass(frozen=True)
class DemoConfig:
    """Runtime configuration for loading PDFs into OracleVectorStore.

    Attributes:
        pdf_dir: Folder containing PDF files to load.
        oracle: Oracle vector store configuration.
        embeddings: OCI embedding provider configuration.
        agent_model: Locus model name used by the RAG agent.
        runtime: Demo runtime behavior configuration.
    """

    pdf_dir: Path
    oracle: OracleStoreConfig
    embeddings: OCIEmbeddingDemoConfig
    agent_model: str
    runtime: RuntimeConfig


def load_config() -> DemoConfig:
    """Load demo configuration from environment variables.

    Returns:
        A populated demo configuration.
    """
    load_dotenv(DEFAULT_ENV_FILE, override=False)

    pdf_dir = _path_from_env("DEMO_RAG_PDF_DIR", "pdf")

    return DemoConfig(
        pdf_dir=pdf_dir,
        oracle=OracleStoreConfig(
            dsn=_optional_env("DEMO_RAG_ORACLE_DSN"),
            user=_env("DEMO_RAG_ORACLE_USER", "admin"),
            password=_env("DEMO_RAG_ORACLE_PASSWORD", ""),
            wallet=OracleWalletConfig(
                location=_optional_env("DEMO_RAG_ORACLE_WALLET_LOCATION"),
                password=_optional_env("DEMO_RAG_ORACLE_WALLET_PASSWORD"),
            ),
            table_name=_env("DEMO_RAG_ORACLE_TABLE_NAME", "locus_demo_rag_vectors"),
            schema_name=_optional_env("DEMO_RAG_ORACLE_SCHEMA_NAME"),
            distance_metric=_env("DEMO_RAG_ORACLE_DISTANCE_METRIC", "COSINE"),
        ),
        embeddings=OCIEmbeddingDemoConfig(
            model_id=_env(
                "DEMO_RAG_OCI_EMBED_MODEL_ID",
                "cohere.embed-multilingual-v3.0",
            ),
            compartment_id=_env("DEMO_RAG_OCI_COMPARTMENT_ID", ""),
            profile_name=_env("DEMO_RAG_OCI_PROFILE_NAME", "DEFAULT"),
            auth_type=_env("DEMO_RAG_OCI_AUTH_TYPE", "api_key"),
            service_endpoint=_optional_env("DEMO_RAG_OCI_SERVICE_ENDPOINT"),
            config_file=_env("DEMO_RAG_OCI_CONFIG_FILE", "~/.oci/config"),
        ),
        agent_model=_env("DEMO_RAG_AGENT_MODEL", "oci:openai.gpt-5.5"),
        runtime=RuntimeConfig(
            chunk_size=_int_env("DEMO_RAG_CHUNK_SIZE", 1000),
            chunk_overlap=_int_env("DEMO_RAG_CHUNK_OVERLAP", 200),
            clear_before_load=_bool_env("DEMO_RAG_CLEAR_BEFORE_LOAD", False),
            log_level=_env("DEMO_RAG_LOG_LEVEL", "INFO"),
        ),
    )


def _env(name: str, default: str) -> str:
    """Return an environment variable value or a default string.

    Args:
        name: Environment variable name.
        default: Value to use when the variable is missing.

    Returns:
        The stripped environment variable value or the default.
    """
    value = os.environ.get(name, default)
    return value.strip() if isinstance(value, str) else default


def _optional_env(name: str) -> str | None:
    """Return a non-empty environment variable value or None.

    Args:
        name: Environment variable name.

    Returns:
        The stripped environment variable value, or None if empty.
    """
    value = os.environ.get(name, "").strip()
    return value or None


def _int_env(name: str, default: int) -> int:
    """Return an integer environment variable value or a default integer.

    Args:
        name: Environment variable name.
        default: Value to use when the variable is missing.

    Returns:
        The parsed integer value or the default.
    """
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def _bool_env(name: str, default: bool) -> bool:
    """Return a boolean environment variable value or a default boolean.

    Args:
        name: Environment variable name.
        default: Value to use when the variable is missing.

    Returns:
        True for common truthy strings, False for other non-empty values,
        or the default when the variable is missing.
    """
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


def _path_from_env(name: str, default: str) -> Path:
    """Return an absolute path from an environment variable or default.

    Args:
        name: Environment variable name.
        default: Relative or absolute default path.

    Returns:
        An absolute path, resolved relative to the repository root when needed.
    """
    raw_value = os.environ.get(name, default).strip()
    path = Path(raw_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path
