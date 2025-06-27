# MedAssist+ RAG Chatbot (TinyLlama 1.1B + FAISS)

A lightweight Retrieval-Augmented Generation chatbot designed to run fully on CPU inside a Hugging Face Space.

## Features
* TinyLlama-1.1B-Chat GGUF quantised model via `ctransformers` (Q4_K_M).
* Sentence-Transformers embeddings + FAISS index for fast passage retrieval.
* Gradio UI (`app.py`) for interactive chat; FastAPI/Gradio backend autostarts on Spaces.
* Automatic first-run download of the GGUF model from the Hub (saves you from pushing a 700 MB file).

## Quick start (local)
```bash
pip install -r requirements.txt
python tinyllama_rag_chatbot.py --build-index corpus.txt  # once
python app.py  # launches Gradio
```

## Deploy to Hugging Face Spaces
1. Create a Space of type **Gradio**.
2. Push this directory (ensure Git-LFS installed). The Space will build automatically.
3. First launch will download the model (~700 MB) – subsequent runs use the cached copy.

---
This information is not a medical diagnosis.
