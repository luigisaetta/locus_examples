"""
Author: L. Saetta
Last update: 2026-05-19
License: MIT
Description: Configuration helpers for the OracleVectorStore RAG demo.
"""

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
        region: OCI GenAI region used when no explicit endpoint is set.
        service_endpoint: Optional OCI GenAI service endpoint.
        config_file: OCI config file path.
    """

    model_id: str
    compartment_id: str
    profile_name: str
    auth_type: str
    region: str
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
class RerankerConfig:
    """Reranker configuration.

    Attributes:
        enabled: Whether to rerank vector search candidates.
        model_id: OCI Cohere rerank model ID.
        top_k: Number of vector search candidates to fetch before reranking.
        top_n: Number of reranked results to keep.
        max_chunks_per_document: Optional OCI rerank chunk limit per document.
        max_tokens_per_document: Optional OCI rerank token limit per document.
    """

    enabled: bool
    model_id: str
    top_k: int
    top_n: int
    max_chunks_per_document: int | None
    max_tokens_per_document: int | None


@dataclass(frozen=True)
class AgentServerConfig:
    """AgentServer configuration.

    Attributes:
        host: Bind host used by the AgentServer.
        port: Bind port used by the AgentServer.
        url: Base URL of the running AgentServer.
    """

    host: str
    port: int
    url: str


@dataclass(frozen=True)
class A2AServerConfig:
    """A2A server configuration.

    Attributes:
        host: Bind host used by the A2A server.
        port: Bind port used by the A2A server.
        url: Public URL advertised in the A2A Agent Card.
        api_key: Optional bearer token required by the A2A server.
    """

    host: str
    port: int
    url: str
    api_key: str | None


@dataclass(frozen=True)
class LangfuseConfig:
    """Langfuse OpenTelemetry export configuration.

    Attributes:
        enabled: Whether Langfuse OpenTelemetry export is enabled.
        endpoint: OTLP HTTP traces endpoint used by the exporter.
        public_key: Langfuse public key for Basic Auth.
        secret_key: Langfuse secret key for Basic Auth.
        service_name: OpenTelemetry service name for emitted spans.
        environment: Deployment environment label for emitted spans.
    """

    enabled: bool
    endpoint: str
    public_key: str | None
    secret_key: str | None
    service_name: str
    environment: str


@dataclass(frozen=True)
class DemoConfig:  # pylint: disable=too-many-instance-attributes
    """Runtime configuration for loading PDFs into OracleVectorStore.

    Attributes:
        pdf_dir: Folder containing PDF files to load.
        oracle: Oracle vector store configuration.
        embeddings: OCI embedding provider configuration.
        reranker: Reranker configuration.
        agent_model: Locus model name used by the RAG agent.
        agent_server: AgentServer configuration.
        a2a_server: A2A server configuration.
        langfuse: Langfuse OpenTelemetry export configuration.
        runtime: Demo runtime behavior configuration.
    """

    pdf_dir: Path
    oracle: OracleStoreConfig
    embeddings: OCIEmbeddingDemoConfig
    reranker: RerankerConfig
    agent_model: str
    agent_server: AgentServerConfig
    a2a_server: A2AServerConfig
    langfuse: LangfuseConfig
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
            region=_env(
                "DEMO_RAG_OCI_REGION",
                os.environ.get("LOCUS_OCI_REGION")
                or os.environ.get("OCI_REGION")
                or "us-chicago-1",
            ),
            service_endpoint=_optional_env("DEMO_RAG_OCI_SERVICE_ENDPOINT"),
            config_file=_env("DEMO_RAG_OCI_CONFIG_FILE", "~/.oci/config"),
        ),
        reranker=RerankerConfig(
            enabled=_bool_env("DEMO_RAG_RERANKER_ENABLED", True),
            model_id=_env("DEMO_RAG_RERANKER_MODEL_ID", "cohere.rerank-v4.0-fast"),
            top_k=_int_env(
                "DEMO_RAG_RETRIEVAL_TOP_K",
                _int_env("DEMO_RAG_RERANKER_CANDIDATE_POOL", 50),
            ),
            top_n=_int_env("DEMO_RAG_RERANKER_TOP_N", 10),
            max_chunks_per_document=_optional_int_env(
                "DEMO_RAG_RERANKER_MAX_CHUNKS_PER_DOCUMENT"
            ),
            max_tokens_per_document=_optional_int_env(
                "DEMO_RAG_RERANKER_MAX_TOKENS_PER_DOCUMENT"
            ),
        ),
        agent_model=_env("DEMO_RAG_AGENT_MODEL", "oci:openai.gpt-5.5"),
        agent_server=AgentServerConfig(
            host=_env("DEMO_RAG_AGENT_SERVER_HOST", "127.0.0.1"),
            port=_int_env("DEMO_RAG_AGENT_SERVER_PORT", 8000),
            url=_env("DEMO_RAG_AGENT_SERVER_URL", "http://127.0.0.1:8000"),
        ),
        a2a_server=A2AServerConfig(
            host=_env("DEMO_RAG_A2A_HOST", "127.0.0.1"),
            port=_int_env("DEMO_RAG_A2A_PORT", 7421),
            url=_env("DEMO_RAG_A2A_URL", "http://127.0.0.1:7421"),
            api_key=_optional_env("DEMO_RAG_A2A_API_KEY"),
        ),
        langfuse=LangfuseConfig(
            enabled=_bool_env("DEMO_RAG_LANGFUSE_ENABLED", False),
            endpoint=_env(
                "DEMO_RAG_LANGFUSE_ENDPOINT",
                "https://cloud.langfuse.com/api/public/otel/v1/traces",
            ),
            public_key=_optional_env("DEMO_RAG_LANGFUSE_PUBLIC_KEY"),
            secret_key=_optional_env("DEMO_RAG_LANGFUSE_SECRET_KEY"),
            service_name=_env(
                "DEMO_RAG_LANGFUSE_SERVICE_NAME",
                "locus-demo-rag-a2a",
            ),
            environment=_env("DEMO_RAG_LANGFUSE_ENVIRONMENT", "local"),
        ),
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


def _optional_int_env(name: str) -> int | None:
    """Return an optional integer environment variable value.

    Args:
        name: Environment variable name.

    Returns:
        The parsed integer value or None when the variable is missing.
    """
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


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
