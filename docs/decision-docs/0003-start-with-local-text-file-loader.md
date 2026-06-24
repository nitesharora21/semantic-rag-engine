# 0003: Starting with a local Text File Loader

## Date

2026-06-23

## Context

The RAG engine needs a way to load source text before it can split and chunk documents.

For now, Im reading a simple document to load and chunk it - will add more functionality later. 

Will add more code to support multi format documents such as pdf, html etc.

## Decision

The first loader will support reading one local text file from the local disk.

The loader accepts:
- A file path

It returns:
- The file content as a string

## Why simple design first?

I want to keep the functionality scoped to a very small capability at this time. Need to build, test and verify that 
the smaller version is working before I start thinking of scaling and adding more functionalities to it.

So Im trying to keep it specific to:
1. Load text
2. Chunk Text
3. Later, embed chunks
4. Later, retreive chunks


The loader is not production ready, but it creates a clear foundation. More coming...
