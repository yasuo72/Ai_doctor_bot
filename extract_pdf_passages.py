#!/usr/bin/env python3
"""Extract passages from one or more PDF files and append them to corpus.txt.

Usage:
    python extract_pdf_passages.py file1.pdf file2.pdf ...

Each paragraph (split by blank lines) is cleaned, truncated to <= 500 chars,
then written as a single line to corpus.txt in the chatbot folder.

Requires:
    pip install pypdf tqdm
"""
from __future__ import annotations
import os, sys, re, textwrap
from pathlib import Path
from typing import List
from tqdm import tqdm
try:
    from pypdf import PdfReader
except ImportError:
    print("Please install pypdf first: pip install pypdf")
    sys.exit(1)

BASE_DIR = Path(__file__).parent
CORPUS_PATH = BASE_DIR / "corpus.txt"
MAX_CHARS = 500


def extract_paragraphs(pdf_path: Path) -> List[str]:
    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    # Normalize spaces
    text = re.sub(r"\s+", " ", text)
    # Split by period followed by space or newline => rough sentences
    raw_pars = re.split(r"\n\s*\n|\.\s", text)
    passages = []
    for para in raw_pars:
        para = para.strip()
        if len(para) < 40:
            continue  # skip very short bits
        # break long paragraphs into chunks
        for chunk in textwrap.wrap(para, MAX_CHARS):
            passages.append(chunk)
    return passages


def main(pdf_paths: List[str]):
    all_passages: List[str] = []
    for p in pdf_paths:
        path = Path(p)
        if not path.exists():
            print(f"Skipping {p}: not found")
            continue
        print(f"Extracting from {p} ...")
        passages = extract_paragraphs(path)
        all_passages.extend(passages)
        print(f" → {len(passages)} passages")

    if not all_passages:
        print("No passages extracted.")
        return

    # Append to corpus.txt
    with open(CORPUS_PATH, "a", encoding="utf-8") as f:
        for passage in all_passages:
            f.write(passage.strip() + "\n")
    print(f"✅ Appended {len(all_passages)} passages to {CORPUS_PATH}")
    print("Remember to rebuild the index:")
    print("  python tinyllama_rag_chatbot.py --build-index corpus.txt")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_pdf_passages.py <pdf1> [pdf2 ...]")
        sys.exit(0)
    main(sys.argv[1:])
