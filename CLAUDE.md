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

## Architecture & App Flow

Everything lives in `app.py`. Here is how the code is structured and what happens when a user asks a question.

### 1. Lazy-loaded singletons (lines 9–40)

Three global objects are created once and reused for every request — this avoids the cost of reconnecting or reloading a model on every question.

| Variable | What it holds | When it's created |
|---|---|---|
| `_supabase` | Connection to the Supabase database | First question asked |
| `_embedder` | The `all-MiniLM-L12-v2` sentence embedding model | First question asked |
| `_llm_client` | HuggingFace `InferenceClient` pointing at Llama 3.1 8B | First question asked |

Each `get_*()` function checks if the global is `None`, creates it if so, and returns it.

### 2. Retrieval — `retrieve_context_from_supabase()` (lines 45–61)

This function finds the most relevant text chunks from the database for a given question:

1. The user's question (a string) is converted into a **vector** (a list of numbers) using the embedding model. Similar sentences produce similar vectors.
2. That vector is sent to Supabase via a stored procedure called `match_pregnancy_chunks`, which uses **pgvector** to find the `k` most similar chunks already stored in the database.
3. The function returns the matching chunks joined as a single `context` string, plus the raw rows for optional debug display.

### 3. Answer generation — `answer_with_llama()` (lines 64–104)

This function takes the user's question, calls the retrieval function above, then asks the LLM to answer:

1. Calls `retrieve_context_from_supabase()` to get the relevant text.
2. Builds a prompt in the format: *"Answer the question based only on the context below. Context: ... Question: ... Answer:"*
3. Sends that prompt to Llama 3.1 8B on HuggingFace. The model is instructed to only use the provided context — it won't make things up from its training data.
4. Returns the answer text, and optionally a debug string showing which chunks were retrieved (source, page range, similarity score).

### 4. Gradio UI (lines 109–128)

Gradio is a Python library that turns functions into a web interface automatically. Here's what each UI element maps to:

| UI element | Variable | Purpose |
|---|---|---|
| Text input box | `question_in` | User types their question here |
| Slider (3–10) | `k_in` | Controls how many chunks to retrieve from the database |
| Checkbox | `show_ctx` | If checked, shows the raw retrieved chunks below the answer |
| "Get answer" button | `btn` | Triggers the call to `answer_with_llama()` |
| Answer box | `answer_out` | Displays the LLM's response |
| Retrieved chunks box | `ctx_out` | Displays debug info when the checkbox is on |

When the user clicks **Get answer**, Gradio calls `answer_with_llama(question, k, show_context)` and puts the two return values into `answer_out` and `ctx_out`.

## Deploying to Hugging Face Spaces

The app is hosted at: `https://huggingface.co/spaces/gurdeep1989/baobaopedia`

### First-time setup

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) — select **Gradio** as the SDK.

2. Set secrets in the Space under **Settings → Variables and Secrets**:
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
   - `HF_TOKEN`

3. Get a HF token with **write** access from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

4. Add the Space as a git remote (with credentials embedded):
   ```bash
   git remote add space https://YOUR_HF_USERNAME:YOUR_HF_TOKEN@huggingface.co/spaces/gurdeep1989/baobaopedia
   ```

5. Push:
   ```bash
   git push space master:main
   ```

### Subsequent pushes

```bash
git add .
git commit -m "your message"
git push space master:main
```

Use `--force` only if you need to overwrite conflicting history on the remote.

## Key External Dependencies

| Dependency | Purpose |
|---|---|
| `supabase` | Vector store client (pgvector via `match_pregnancy_chunks` RPC) |
| `sentence-transformers` | Embedding model for query vectorisation |
| `huggingface_hub` | LLM inference (`InferenceClient`) |
| `gradio` | Web UI framework |
| `numpy` | Embedding array handling |
