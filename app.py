"""Gradio / FastAPI wrapper for TinyLlama RAG so it can run in a Hugging Face Space.
The heavy lifting stays in tinyllama_rag_chatbot.py – we just call its helpers.
"""
from __future__ import annotations

import gradio as gr

# Re-use retrieval & LLM helpers from the CLI script
from tinyllama_rag_chatbot import retrieve, llm, PROMPT  # type: ignore

# ‑-- Simple one-shot chat helper -------------------------------------------------

def chat_once(question: str) -> str:
    """Return one answer given a user question."""
    if not question.strip():
        return "Please enter a question."

    # Retrieve supporting passages
    ctx_passages = retrieve(question)
    # Take first 250 chars of each passage, clean newlines
    snippets = [p.replace("\n", " ")[:250].strip() for p in ctx_passages]
    raw_context = "\n---\n".join(snippets)
    # Cap overall context length (~900 chars ≈ 225 tokens)
    context = raw_context[:900]

    prompt = PROMPT.format(context=context, question=question)
    response = llm()(prompt, max_new_tokens=120, temperature=0.1)
    return response.strip()


# ‑-- Gradio UI -------------------------------------------------------------------

def gradio_ui():
    """Launch a minimal Gradio Blocks interface."""
    with gr.Blocks(title="MedAssist+ (TinyLlama + RAG)") as demo:
        gr.Markdown("""# MedAssist+  \nOffline TinyLlama-1.1B Chat + Retrieval (FAISS).  \\
**Disclaimer:** This information is not a medical diagnosis.""")
        inp = gr.Textbox(label="Your question", placeholder="e.g. How is HER2-positive breast cancer treated?", lines=2)
        out = gr.Markdown(label="Answer")

        inp.submit(fn=chat_once, inputs=inp, outputs=out)

    return demo


# Create global Blocks object for HF Spaces autolaunch
# (HF searches for `demo`, `app`, or `interface` at import time.)
demo = gradio_ui()

def main():  # Fallback CLI entry point
    ui = gradio_ui()
    # On Spaces we set share=False; HF handles routing
    ui.launch(show_error=True, share=False)


if __name__ == "__main__":
    main()
