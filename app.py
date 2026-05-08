import os
import numpy as np
import gradio as gr
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient


# ------------------ Cached singletons (efficient) ------------------

_supabase: Client | None = None
_embedder: SentenceTransformer | None = None
_llm_client: InferenceClient | None = None


def get_supabase_client() -> Client:
    global _supabase
    if _supabase is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_KEY"]
        _supabase = create_client(url, key)
    return _supabase


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    return _embedder


def get_llm_client() -> InferenceClient:
    global _llm_client
    if _llm_client is None:
        hf_token = os.environ["HF_TOKEN"]
        _llm_client = InferenceClient(
            model="meta-llama/Llama-3.1-8B-Instruct",
            token=hf_token,
        )
    return _llm_client


# ------------------ Core RAG functions ------------------

def retrieve_context_from_supabase(question: str, k: int = 5):
    supabase = get_supabase_client()
    embedder = get_embedder()

    q_vec = embedder.encode([question], normalize_embeddings=True).astype(np.float32)[0]

    resp = supabase.rpc(
        "match_pregnancy_chunks",
        {
            "query_embedding": q_vec.tolist(),
            "match_count": int(k),
        },
    ).execute()

    rows = resp.data or []
    context = "\n\n".join([r.get("content", "") for r in rows])
    return context, rows


def answer_with_llama(question: str, k: int = 5, show_context: bool = False):
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", ""

    context, rows = retrieve_context_from_supabase(question, k=k)

    if not context.strip():
        return "I couldn't find relevant context in the knowledge base.", ""

    llm_client = get_llm_client()

    prompt = (
        "Answer the question based only on the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    response = llm_client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who answers based only on the given context."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        temperature=0.3,
    )

    answer = response.choices[0].message["content"]

    # Optional: display retrieved chunks for transparency/debugging
    if show_context:
        formatted = []
        for i, r in enumerate(rows, start=1):
            meta = f"Chunk {i} | source={r.get('source')} | pages={r.get('page_range')} | idx={r.get('chunk_index')} | sim={r.get('similarity')}"
            formatted.append(meta + "\n" + (r.get("content") or ""))
        debug_text = "\n\n" + ("-" * 60) + "\n\n".join(formatted)
    else:
        debug_text = ""

    return answer, debug_text


# ------------------ Gradio UI ------------------

with gr.Blocks(title="Pregnancy Book Q&A (RAG POC)") as demo:
    gr.Markdown("## 👶 Pregnancy Book Q&A (RAG POC)\nAsk a question. The app retrieves the most relevant chunks from Supabase (pgvector) and answers using Llama 3.1 8B on Hugging Face.")

    with gr.Row():
        question_in = gr.Textbox(label="Your question", value="What are pain interventions?", lines=2)
    with gr.Row():
        k_in = gr.Slider(minimum=3, maximum=10, value=5, step=1, label="Number of chunks to retrieve (top-k)")
        show_ctx = gr.Checkbox(label="Show retrieved chunks (debug)", value=False)

    btn = gr.Button("Get answer")

    answer_out = gr.Textbox(label="Answer", lines=8)
    ctx_out = gr.Textbox(label="Retrieved chunks", lines=16, visible=True)

    btn.click(
        fn=answer_with_llama,
        inputs=[question_in, k_in, show_ctx],
        outputs=[answer_out, ctx_out],
    )

demo.launch()