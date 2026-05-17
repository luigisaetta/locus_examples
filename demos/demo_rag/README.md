# RAG Demo with Locus and OracleVectorStore

This demo loads PDF files from the repository root `pdf` folder into an Oracle
vector store managed by `locus.rag.OracleVectorStore`.

## Run

From the repository root:

```bash
conda install -n locus_examples -c conda-forge oracledb pypdf
source demos/demo_rag/.env
python -m demos.demo_rag.load_pdfs
```

To start the RAG agent server:

```bash
source demos/demo_rag/.env
python -m demos.demo_rag.serve_agent
```

To query the running server:

```bash
source demos/demo_rag/.env
python -m demos.demo_rag.query_agent "What does the knowledge base say about ...?"
```

## Note

- `OracleVectorStore` creates the Oracle table automatically on the first store
  operation.
- The demo default table name is `locus_demo_rag_vectors`.
- The agent model is configured with `DEMO_RAG_AGENT_MODEL`.
- The query client uses `DEMO_RAG_AGENT_SERVER_URL`.
- PDF text extraction requires `pypdf` or `PyPDF2` in the conda environment.
- Oracle connectivity requires `oracledb` in the conda environment.
- Runtime configuration is read from `demos/demo_rag/.env`; use
  `demos/demo_rag/.env.example` as the committed template.
