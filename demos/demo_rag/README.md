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

## Note

- `OracleVectorStore` creates the Oracle table automatically on the first store
  operation.
- The demo default table name is `locus_demo_rag_vectors`.
- PDF text extraction requires `pypdf` or `PyPDF2` in the conda environment.
- Oracle connectivity requires `oracledb` in the conda environment.
- Runtime configuration is read from `demos/demo_rag/.env`; use
  `demos/demo_rag/.env.example` as the committed template.
