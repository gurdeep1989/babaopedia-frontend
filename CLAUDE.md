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

### 3. Answer generation — `answer_with_llama(question, history, k)`

This function is the core of the app. It is called automatically by Gradio every time the user sends a message.

**Parameters:**
- `question` — the latest message the user typed
- `history` — all previous turns in the current session, passed automatically by Gradio
- `k` — number of chunks to retrieve (comes from the slider, default 5)

**What it does:**
1. Calls `retrieve_context_from_supabase()` to fetch the most relevant text chunks for the current question.
2. Builds the `messages` list for the LLM:
   - Starts with a system prompt telling Llama to only answer from the given context.
   - Appends all previous conversation turns from `history` so Llama can handle follow-up questions.
   - Appends the current question wrapped with the retrieved context.
3. Calls Llama 3.1 8B via HuggingFace `InferenceClient` and returns the answer string.

**History format compatibility:** Gradio 4 passes history as a list of `[user, assistant]` tuples; Gradio 5 passes it as a list of `{"role": ..., "content": ...}` dicts. The code handles both formats with an `isinstance(item, dict)` check.

### 4. Gradio ChatInterface

Gradio is a Python library that turns a Python function into a web UI automatically. The app uses `gr.ChatInterface`, which is Gradio's built-in chat component — it renders a full chat window, handles the message input box, send button, and conversation history automatically.

**How `gr.ChatInterface` works:**
- The user types a message and presses Enter (or clicks the send button).
- Gradio calls `answer_with_llama(question, history, k)` automatically, passing the current message, the full conversation history so far, and the value of the slider.
- The returned string is displayed as the assistant's reply and added to the history for the next turn.
- The conversation history lives in the browser session — refreshing the page starts a fresh conversation.

**UI elements:**

| Element | Purpose |
|---|---|
| Chat window | Displays the full conversation |
| Text input (bottom) | User types their question; press Enter or click → to send |
| "Additional inputs" panel | Expands to reveal the top-k slider |
| Slider (3–10) | Controls how many chunks to retrieve from Supabase per question |

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

### Deploying changes

Every time you change code locally, push it to the Space with:

```bash
git add app.py                        # stage the changed file(s)
git commit -m "describe what changed"
git push space master:main            # deploy to HuggingFace
```

After pushing, HuggingFace automatically detects the new commit, reinstalls dependencies if `requirements.txt` changed, and restarts the app. Watch the **Logs** tab in your Space to see the build progress.

Use `--force` only if you need to overwrite conflicting history on the remote:
```bash
git push --force space master:main
```

### Checking what's deployed

If you're unsure whether HF has your latest code, run:
```bash
git log space/main --oneline -3
```
Compare the top commit hash against your local `git log --oneline -3`. If they differ, you need to push.

## Key External Dependencies

| Dependency | Purpose |
|---|---|
| `supabase` | Vector store client (pgvector via `match_pregnancy_chunks` RPC) |
| `sentence-transformers` | Embedding model for query vectorisation |
| `huggingface_hub` | LLM inference (`InferenceClient`) |
| `gradio` | Web UI framework |
| `numpy` | Embedding array handling |
