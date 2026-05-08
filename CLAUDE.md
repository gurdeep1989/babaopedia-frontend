# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a RAG (Retrieval-Augmented Generation) proof-of-concept called **Babaopedia** — a Q&A app for a pregnancy book. It uses a Gradio web UI, retrieves relevant text chunks from Supabase (pgvector), and generates answers via Llama 3.1 8B on Hugging Face Inference API.

## Running the App

Install dependencies:
```bash
pip install -r requirements.txt
```

Set required environment variables before running:
```bash
export SUPABASE_URL=<your-supabase-url>
export SUPABASE_KEY=<your-supabase-key>
export HF_TOKEN=<your-huggingface-token>
```

Run the app:
```bash
python app.py
```

Gradio will start a local server (default: http://127.0.0.1:7860).

## Architecture

Everything lives in `app.py`. There are three layers:

1. **Singletons** — `get_supabase_client()`, `get_embedder()`, `get_llm_client()` are lazily initialized globals. The embedder uses `sentence-transformers/all-MiniLM-L12-v2`; the LLM client points to `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace `InferenceClient`.

2. **RAG pipeline** — `retrieve_context_from_supabase()` encodes the user question into a vector and calls the `match_pregnancy_chunks` Supabase RPC function (pgvector similarity search). `answer_with_llama()` builds a context-only prompt and calls the LLM.

3. **Gradio UI** — a `gr.Blocks` interface with a textbox for the question, a slider for top-k chunks, a debug checkbox to show retrieved chunks, and two output textboxes.

## Key External Dependencies

| Dependency | Purpose |
|---|---|
| `supabase` | Vector store client (pgvector via `match_pregnancy_chunks` RPC) |
| `sentence-transformers` | Embedding model for query vectorisation |
| `huggingface_hub` | LLM inference (`InferenceClient`) |
| `gradio` | Web UI framework |
| `numpy` | Embedding array handling |
