# Demo RAG con Locus e OracleVectorStore

Questa demo carica i PDF presenti nella cartella `pdf` della root del repository in un vector store Oracle gestito da `locus.rag.OracleVectorStore`.

## Esecuzione

Da root del repository:

```bash
conda install -n locus_examples -c conda-forge oracledb pypdf
source demos/demo_rag/.env
python -m demos.demo_rag.load_pdfs
```

## Note

- La tabella Oracle viene creata automaticamente da `OracleVectorStore` alla prima operazione sullo store.
- Il nome tabella di default della demo è `locus_demo_rag_vectors`.
- Per estrarre testo dai PDF serve `pypdf` oppure `PyPDF2` installato nell'ambiente conda.
- Per collegarsi a Oracle serve il pacchetto `oracledb` installato nell'ambiente conda.
