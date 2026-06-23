# 0001: Architecture Decision Records

## Date

2026-06-22

## Context

This project is intended to build a practicel Retreival-Augmented Generation system for answering questions over trusted knowledge sources.

A RAG system involves many design choices, some of them are:
- How documents are loaded
- How text is chunked
- Which embedding model is used
- Which vector store is used
- How retreival is evaluated
- How generated answers are grounded in retreived context
- How unsupported or low-confidence answers are handled

Im going to build this project incrementally and will document decisions, and the reasons why..

## Decision

This project will use Architecture Decision Records stored inside  `docs/decisions/`

Each ADR will describe:
- The problem or design choice
- The context behind the decision
- The selected approach
- Alternatives considered
- Tradeoffs and consequences




