#!/usr/bin/env python3
"""MedAssist+ – TinyLlama + Retrieval (fully-offline)

Requirements (add to requirements.txt):
    ctransformers>=0.3.0
    sentence-transformers>=2.6.1
    faiss-cpu>=1.7.4
    tqdm

Model files (download once):
1. TinyLlama-1.1B-Chat GGUF Q4_0
   https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-GGUF
   place as  `models/tinyllama-1.1b-chat.Q4_0.gguf`
2. Embedding model – we use a very small one to keep RAM low:
   sentence-transformers/paraphrase-MiniLM-L6-v2  (automatic download)

Index build (first run only):
    python tinyllama_rag_chatbot.py --build-index corpus.txt
    # corpus.txt should be a UTF-8 text file with one passage per line.
    # You can generate it from PDFs using any extractor and simple sentence splitting.

Chat mode:
    python tinyllama_rag_chatbot.py
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import List

import faiss
from tqdm import tqdm
from huggingface_hub import hf_hub_download

INDEX_PATH = os.path.join(os.path.dirname(__file__), "rag_index.faiss")
META_PATH = os.path.join(os.path.dirname(__file__), "rag_meta.pkl")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# ---------------- Embeddings -----------------
from sentence_transformers import SentenceTransformer
EMBED_MODEL_NAME = "paraphrase-MiniLM-L6-v2"  # ~120 MB, 384-dim
_embedder: SentenceTransformer | None = None

def embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def build_index(passages: List[str]):
    vecs = embedder().encode(passages, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(passages, f)
    print(f"✅ Built index with {len(passages)} passages → {INDEX_PATH}")

# ---------------- LLM -----------------
from ctransformers import AutoModelForCausalLM

_llm = None

def llm():
    global _llm
    if _llm is None:
        model_path = MODEL_PATH
        if not os.path.exists(model_path):
            # Download quantised TinyLlama model (~700 MB) on first run
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print("⏬ Downloading TinyLlama GGUF model from HF Hub...")
            model_path = hf_hub_download(
                repo_id="TheBloke/TinyLlama-1.1B-Chat-GGUF",
                filename=os.path.basename(model_path),
                revision="main",
                local_dir=os.path.dirname(model_path),
            )
            print("✅ Model stored at", model_path)

        _llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="llama",
            gpu_layers=0,   # CPU-only; set >0 if GPU available
        )
    return _llm

# ---------------- Retrieval helper -----------------

def retrieve(query: str, top_k: int = 12):
    if not os.path.exists(INDEX_PATH):
        print("❌ Retrieval index not found. Build it with --build-index <file> first.")
        sys.exit(1)
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        passages = pickle.load(f)
    qvec = embedder().encode([query], normalize_embeddings=True)
    D, I = index.search(qvec, top_k)
    return [passages[idx] for idx in I[0]]

# ---------------- Prompt template -----------------
PROMPT = """You are MedAssist+, a cautious medical information assistant.
Answer ONLY with facts you can find in the CONTEXT below. If the
information is insufficient, reply exactly:
"I am not certain based on the provided information. Please consult a physician."
Always append the line: "This information is not a medical diagnosis."
---
CONTEXT:\n{context}\n---
QUESTION: {question}\n
ANSWER:"""

# ---------------- Main -----------------

def chat():
    print("=== MedAssist+ (TinyLlama + RAG) ===")
    print("Type your question. 'exit' to quit.\n")
    while True:
        try:
            user_q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if user_q.lower() in {"exit", "quit", "q"}:
            break
        if not user_q:
            continue

        ctx_passages = retrieve(user_q)
        # Take first 150 characters of each passage to keep diversity
        snippets = [p.replace("\n", " ")[:250].strip() for p in ctx_passages]
        raw_context = "\n---\n".join(snippets)
        # Cap overall context to ~600 characters (≈150 tokens)
        context = raw_context[:700]
        prompt = PROMPT.format(context=context, question=user_q)
        response = llm()(prompt, max_new_tokens=96, temperature=0.1)
        print(response)
        print()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--build-index", type=str, metavar="TXT", help="Path to text file with one passage per line.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.build_index:
        if not os.path.exists(args.build_index):
            print("File not found:", args.build_index)
            return
        with open(args.build_index, "r", encoding="utf-8") as f:
            passages = [line.strip() for line in f if line.strip()]
        build_index(passages)
    else:
        chat()


if __name__ == "__main__":
    main()
