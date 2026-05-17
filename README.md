# locus_examples

This repository contains small, runnable examples and demos for exploring the
Oracle Locus framework. The goal is to keep each example focused, easy to run
from the repository root, and useful as a reference for agents, multi-agent
flows, RAG pipelines, Oracle-backed vector stores, HTTP serving, and A2A
interoperability.

Use the `locus_examples` conda environment for local commands.

## Demos

| Demo | Description |
| --- | --- |
| [RAG with OracleVectorStore](demos/demo_rag) | End-to-end RAG demo using Locus, OCI embeddings, and `OracleVectorStore`. It loads PDFs from a local `pdf/` folder, stores chunks in Oracle Database native vector storage, serves a RAG-enabled Locus agent over HTTP, and includes CLI clients for querying the server. **This demo also exposes the same search agent through an A2A server** with an Agent Card and an A2A query client. |

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
- Run Black and Pylint before considering Python changes complete.
