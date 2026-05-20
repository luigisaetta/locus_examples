# Locus Examples

![Python >= 3.11](https://img.shields.io/badge/python-%3E%3D3.11-blue)
![Code style: Black](https://img.shields.io/badge/code%20style-black-000000)
![Lint: Pylint](https://img.shields.io/badge/lint-pylint-yellow)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

This repository contains small, runnable examples and demos for exploring the
[Oracle Locus](https://oracle-samples.github.io/locus/). The goal is to keep
each example focused, easy to run from the repository root, and useful as a
reference for agents, multi-agent flows, RAG pipelines, Oracle-backed vector
stores, HTTP serving, and A2A interoperability.

Use the `locus_examples` conda environment for local commands.

## Demos

| Demo | Description |
| --- | --- |
| [RAG with OracleVectorStore](demos/demo_rag) | End-to-end RAG demo using [Oracle Locus](https://oracle-samples.github.io/locus/), OCI embeddings, and `OracleVectorStore`. It loads PDFs from a local `pdf/` folder, stores chunks in Oracle Database native vector storage, serves a RAG-enabled Locus agent over HTTP, and includes CLI clients for querying the server. **This demo also exposes the same search agent through an A2A server** with an Agent Card and an A2A query client. |

## Examples

| Example | Description |
| --- | --- |
| [example01.py](examples/example01.py) | Minimal synchronous `Agent` call. |
| [example02.py](examples/example02.py) | Streaming agent execution with Locus events. |
| [example03.py](examples/example03.py) | Function-based `StateGraph` flow with accumulated state. |
| [example04.py](examples/example04.py) | Class-based callable nodes inside a `StateGraph`. |
| [example05.py](examples/example05.py) | Exposes a `StateGraph` through `AgentServer` using a small adapter. |

## Development Notes

- Keep documentation and README files in English.
- Keep changes atomic and simple.
- Run Black, Pylint, and pytest before considering Python changes complete.
- Add or update pytest tests whenever introducing new behavior.

## Tests

Install pytest in the `locus_examples` conda environment, then run the suite
from the repository root:

```bash
conda activate locus_examples
python -m pytest
```
