# Repository Guidelines

## Purpose

This repository contains examples and small demos for exploring the Oracle Locus
framework, with a focus on agents, multi-agent flows, RAG, and Oracle-backed
vector stores.

## Operating Notes

- Always use the `locus_examples` conda environment for local commands.
- Before considering a code change complete, run Black and Pylint on the touched
  Python code and fix all reported issues.
- Keep documentation and README files in English.
- Keep changes atomic, scoped, and as simple as possible. Do not over-engineer.
- Prefer demos that can be run from the repository root with `python -m ...`.
- Keep demo configuration in dedicated `.env.example` files. Real `.env` files
  should remain local and uncommitted.
- When the working tree contains unrelated changes, stage and commit only the
  files that belong to the current task.
- Use Google-style docstrings for new demo functions and classes.
- Every Python file must start with an English multiline string header
  surrounded by triple double quotes (`"""`). `Author: L. Saetta` must start
  on the line after the opening triple double quotes, followed by
  `Last update: <YYYY-MM-DD>`, `License: MIT`, and
  `Description: <brief description>`.
