# RAG Demo with Locus and OracleVectorStore

This demo loads PDF files from the repository root `pdf` folder into an Oracle
vector store managed by `locus.rag.OracleVectorStore`. Retrieval uses a
retrieve-then-rerank flow: semantic vector search first fetches a wider
candidate set, then an OCI Cohere reranker reorders those chunks before the
agent sees them.

## Reranking

The reranker is enabled by default and uses `cohere.rerank-v4.0-fast`.

```bash
DEMO_RAG_RERANKER_ENABLED=true
DEMO_RAG_RERANKER_MODEL_ID=cohere.rerank-v4.0-fast
DEMO_RAG_RETRIEVAL_TOP_K=50
DEMO_RAG_RERANKER_TOP_N=10
```

`DEMO_RAG_RETRIEVAL_TOP_K` controls how many candidates are fetched from
OracleVectorStore before reranking. `DEMO_RAG_RERANKER_TOP_N` controls how many
reranked chunks are returned to the agent. Set
`DEMO_RAG_RERANKER_ENABLED=false` to keep the original semantic-only ordering.

## Run

Run every command from the repository root.

Install the demo runtime dependencies once:

```bash
conda install -n locus_examples -c conda-forge oracledb pypdf
pip install -r demos/demo_rag/requirements.txt
```

## Langfuse tracing

The A2A and HTTP agents can export Locus invocation, iteration, and tool-call
spans to Langfuse through OpenTelemetry. The demo uses a custom hook so
`tool.search_knowledge` spans are children of the same `agent.invocation` trace.

Enable tracing in `demos/demo_rag/.env`:

```bash
export DEMO_RAG_LANGFUSE_ENABLED=true
export DEMO_RAG_LANGFUSE_ENDPOINT=https://cloud.langfuse.com/api/public/otel/v1/traces
export DEMO_RAG_LANGFUSE_PUBLIC_KEY=pk-lf-...
export DEMO_RAG_LANGFUSE_SECRET_KEY=sk-lf-...
export DEMO_RAG_LANGFUSE_SERVICE_NAME=locus-demo-rag-a2a
export DEMO_RAG_LANGFUSE_ENVIRONMENT=local
```

The hook records operational metadata only. Tool arguments, retrieved chunks,
and final results are not exported as span attributes.

Load PDF files from the root `pdf` folder into OracleVectorStore:

```bash
source demos/demo_rag/.env
python -m demos.demo_rag.load_pdfs
```

Start the HTTP AgentServer in one terminal:

```bash
source demos/demo_rag/.env
python -m demos.demo_rag.serve_agent
```

Query the running HTTP AgentServer from another terminal:

```bash
source demos/demo_rag/.env
python -m demos.demo_rag.query_agent "What does the knowledge base say about ...?"
```

Start the A2A server in one terminal:

```bash
source demos/demo_rag/.env
python -m demos.demo_rag.serve_a2a
```

Query the running A2A server from another terminal:

```bash
source demos/demo_rag/.env
python -m demos.demo_rag.query_a2a "What does the knowledge base say about ...?"
```

## Note

- `OracleVectorStore` creates the Oracle table automatically on the first store
  operation.
- The demo default table name is `locus_demo_rag_vectors`.
- The agent model is configured with `DEMO_RAG_AGENT_MODEL`.
- The AgentServer uses `DEMO_RAG_AGENT_SERVER_HOST` and
  `DEMO_RAG_AGENT_SERVER_PORT`; the query client uses
  `DEMO_RAG_AGENT_SERVER_URL`.
- The A2A server uses `DEMO_RAG_A2A_HOST`, `DEMO_RAG_A2A_PORT`,
  `DEMO_RAG_A2A_URL`, and `DEMO_RAG_A2A_API_KEY`.
- Langfuse tracing uses `DEMO_RAG_LANGFUSE_ENABLED`,
  `DEMO_RAG_LANGFUSE_ENDPOINT`, `DEMO_RAG_LANGFUSE_PUBLIC_KEY`,
  `DEMO_RAG_LANGFUSE_SECRET_KEY`, `DEMO_RAG_LANGFUSE_SERVICE_NAME`, and
  `DEMO_RAG_LANGFUSE_ENVIRONMENT`.
- PDF text extraction requires `pypdf` or `PyPDF2` in the conda environment.
- Oracle connectivity requires `oracledb` in the conda environment.
- Runtime configuration is read from `demos/demo_rag/.env`; use
  `demos/demo_rag/.env.example` as the committed template.
