# Cell-1

import sys, os, subprocess

CUSTOM_SITE = "/content/custom_site_packages"
os.makedirs(CUSTOM_SITE, exist_ok=True)

if CUSTOM_SITE not in sys.path:
    sys.path.insert(0, CUSTOM_SITE)
os.environ["PYTHONPATH"] = CUSTOM_SITE + ":" + os.environ.get("PYTHONPATH", "")

print("Using custom site-packages:", CUSTOM_SITE)

subprocess.run(
    [
        sys.executable, "-m", "pip", "install",
        "--target", CUSTOM_SITE,
        "-q",
        "google-generativeai",
        "faiss-cpu",
        "fastapi",
        "uvicorn",
        "nest_asyncio",
        "transformers",
        "datasets",
        "sentence-transformers",
        "peft",
        "accelerate",
        "pydantic==2.7.4",
        "aiohttp",
        "anyio>=4"
    ],
    check=True
)

print("✔️ All packages installed in custom_site_packages.")
print("sys.path[0] =", sys.path[0])

# Cell-2

import os, sys, json, re, time
from pathlib import Path
from typing import List, Dict, Any

CUSTOM_SITE = "/content/custom_site_packages"
if CUSTOM_SITE not in sys.path:
    sys.path.insert(0, CUSTOM_SITE)
os.environ["PYTHONPATH"] = CUSTOM_SITE + ":" + os.environ.get("PYTHONPATH", "")

print("sys.path[0] =", sys.path[0])

import numpy as np
import torch
import requests

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model

from fastapi import FastAPI, HTTPException
import uvicorn, nest_asyncio
from pyngrok import ngrok

os.environ["GEMINI_API_KEY"] = "AIzaSyDWBrl1REtSeqPXD6KefEZk-2RDiTsTH9Y"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Please paste your Gemini API key into os.environ['GEMINI_API_KEY'].")

GEMINI_MODEL = "gemini-2.5-flash"

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

print("Gemini REST endpoint:", GEMINI_API_URL)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLASSIFICATION_DATASET = "hate_speech_offensive"
CLASSIFICATION_MODEL = "distilbert-base-uncased"

print("\nConfig:")
print(" EMBED_MODEL:", EMBED_MODEL)
print(" CLASSIFICATION_MODEL:", CLASSIFICATION_MODEL)
print(" GPU available:", torch.cuda.is_available())

# Cell-3

import os

os.environ["GEMINI_API_KEY"] = "AIzaSyDWBrl1REtSeqPXD6KefEZk-2RDiTsTH9Y"

print("GEMINI_API_KEY set?", bool(os.environ.get("GEMINI_API_KEY")))

# Cell-4

def call_gemini(prompt: str, max_output_tokens: int = 256, temperature: float = 0.2) -> str:
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }

    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }

    resp = requests.post(GEMINI_API_URL, headers=headers, json=body, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")

    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        text = json.dumps(data, indent=2)
    return text

# A) Select an open-source pre-trained conversational language model of your choice (that you can take, e.g., from the Hugging Face Transformers library). 
    # Implement a RAG pipeline to enhance responses with external knowledge or your choice. [10 points]

# Cell-5

embedder = SentenceTransformer(EMBED_MODEL)

wiki_stream = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True
)

MAX_ARTICLES = 500
docs = []
count = 0

for item in wiki_stream:
    if count >= MAX_ARTICLES:
        break
    title = item.get("title", f"doc_{count}")
    text = item.get("text", "")
    for j, para in enumerate(text.split("\n\n")):
        para = para.strip()
        if len(para) > 200:
            docs.append({"id": f"{count}_{j}", "title": title, "text": para})
    count += 1

print("Prepared", len(docs), "paragraph-level documents for RAG.")

# Cell-6

texts = [d["text"] for d in docs]
corpus_embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
embedding_dim = corpus_embeddings.shape[1]

faiss.normalize_L2(corpus_embeddings)
index = faiss.IndexFlatIP(embedding_dim)
index.add(corpus_embeddings)

print("FAISS index built with", index.ntotal, "vectors.")

# Cell-7

def retrieve(query: str, k: int = 5):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, k)
    passages = [docs[int(idx)] for idx in idxs[0]]
    return passages, scores[0]

def build_prompt(query: str, passages: List[Dict[str, Any]]) -> str:
    ctx = "\n\n".join([f"Source: {p['title']}\n{p['text']}" for p in passages])
    return (
        "You are a helpful assistant. Use the following context to answer the question. "
        "Cite the source titles when relevant.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
    )

def rag_answer_gemini(query: str, k: int = 5):
    passages, scores = retrieve(query, k=k)
    prompt = build_prompt(query, passages)
    answer = call_gemini(prompt, max_output_tokens=300, temperature=0.0)
    return {
        "answer": answer,
        "retrieved_titles": [p["title"] for p in passages],
        "scores": scores.tolist(),
    }

# Cell-8

res = rag_answer_gemini(
    "What is an algorithm and why are algorithms important in computer science?",
    k=5
)
print("Answer:\n", res["answer"])
print("\nRetrieved titles:", res["retrieved_titles"])
