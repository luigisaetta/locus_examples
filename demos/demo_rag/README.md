# RAG Demo with Locus and OracleVectorStore

This demo loads PDF files from the repository root `pdf` folder into an Oracle
vector store managed by `locus.rag.OracleVectorStore`.

## Run

Run every command from the repository root.

Install the demo runtime dependencies once:

```bash
conda install -n locus_examples -c conda-forge oracledb pypdf
```

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

Fetch the A2A Agent Card and query the running A2A server from another terminal:

```bash
source demos/demo_rag/.env
python -m demos.demo_rag.query_a2a "What does the knowledge base say about ...?"
```

## Note

- `OracleVectorStore` creates the Oracle table automatically on the first store
  operation.
- The demo default table name is `locus_demo_rag_vectors`.
- The agent model is configured with `DEMO_RAG_AGENT_MODEL`.
- The query client uses `DEMO_RAG_AGENT_SERVER_URL`.
- The A2A server uses `DEMO_RAG_A2A_HOST`, `DEMO_RAG_A2A_PORT`,
  `DEMO_RAG_A2A_URL`, and `DEMO_RAG_A2A_API_KEY`.
- PDF text extraction requires `pypdf` or `PyPDF2` in the conda environment.
- Oracle connectivity requires `oracledb` in the conda environment.
- Runtime configuration is read from `demos/demo_rag/.env`; use
  `demos/demo_rag/.env.example` as the committed template.
