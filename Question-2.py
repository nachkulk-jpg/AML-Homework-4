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

# B) Select an open-source pre-trained conversational language model of your choice (that you can take, e.g., from the Hugging Face Transformers library). 
# Use a small domain-specific dataset to fine-tune the model (or use parameter-efficient fine-tuning methods like LoRA or QLoRA). [10 points]

# Cell-9

raw = load_dataset(CLASSIFICATION_DATASET)
print("Original splits:", raw.keys())

full = raw["train"].train_test_split(test_size=0.2, seed=42)
train_raw = full["train"].shuffle(seed=42).select(range(2000))
test_raw  = full["test"].shuffle(seed=42).select(range(500))

print("New splits: train =", len(train_raw), ", test =", len(test_raw))
print("Example row:", train_raw[0])

# Cell-10

tokenizer_cls = AutoTokenizer.from_pretrained(CLASSIFICATION_MODEL)

print("Train columns:", train_raw.column_names)

if "label" in train_raw.column_names:
    LABEL_COL = "label"
elif "class" in train_raw.column_names:
    LABEL_COL = "class"
else:
    raise ValueError(f"Could not find a label column in {train_raw.column_names}")

print("Using label column:", LABEL_COL)

def preprocess(examples):
    return tokenizer_cls(
        examples["tweet"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

train_tok = train_raw.map(preprocess, batched=True)
test_tok  = test_raw.map(preprocess, batched=True)

print("Tokenized train columns:", train_tok.column_names)

# Cell-11

num_labels = len(set(train_raw[LABEL_COL]))
base_model = AutoModelForSequenceClassification.from_pretrained(
    CLASSIFICATION_MODEL,
    num_labels=num_labels,
)

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_lin", "k_lin", "v_lin"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
)

model_peft = get_peft_model(base_model, lora_cfg)

trainable = sum(p.numel() for p in model_peft.parameters() if p.requires_grad)
total = sum(p.numel() for p in model_peft.parameters())
print(f"Trainable params: {trainable} / {total} ({100*trainable/total:.3f}%)")

# Cell-12

print("train_tok columns BEFORE rename:", train_tok.column_names)
print("test_tok  columns BEFORE rename:", test_tok.column_names)

if "labels" in train_tok.column_names:
    effective_label_col = "labels"
else:
    if LABEL_COL not in train_tok.column_names:
        raise ValueError(
            f"Original label column {LABEL_COL!r} not in dataset. "
            f"Current columns: {train_tok.column_names}"
        )
    train_tok = train_tok.rename_column(LABEL_COL, "labels")
    test_tok  = test_tok.rename_column(LABEL_COL, "labels")
    effective_label_col = "labels"

print("Using label column:", effective_label_col)
print("train_tok columns AFTER rename:", train_tok.column_names)
print("test_tok  columns AFTER rename:", test_tok.column_names)

train_tok.set_format(type="torch", columns=["input_ids", "attention_mask", effective_label_col])
test_tok.set_format(type="torch",  columns=["input_ids", "attention_mask", effective_label_col])

training_args = TrainingArguments(
    output_dir="./distilbert_lora_output",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    report_to="none"
)

trainer = Trainer(
    model=model_peft,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=test_tok,
)

print("Trainer ready. Next: run trainer.train() in Cell B5 when you're ready.")

# Cell-13

import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_PROJECT"] = "ignore"
os.environ["WANDB_START_METHOD"] = "thread"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

trainer.train()
trainer.save_model("./distilbert_lora_output")

# Cell-14

from torch.utils.data import DataLoader
import numpy as np
import torch

def eval_subset(model, dataset, n=200):
    subset = dataset.select(range(min(n, len(dataset))))
    loader = DataLoader(subset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y_true = batch["labels"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()
            y_pred = logits.argmax(axis=-1)

            preds.append(y_pred)
            labels.append(y_true)

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    acc = (preds == labels).mean()
    return acc

try:
    trained_model = trainer.model
except NameError:
    raise RuntimeError("Trainer (and its model) is not defined. Run B4 and B5 first.")

accuracy = eval_subset(trained_model, test_tok, n=200)
print(f"Accuracy on 200-sample subset: {accuracy:.4f}")

# C) Develop and test a Model Context Protocol (MCP) Server in Python that bridges an AI client to an external data source or utility. 
# You should test your server using an LLM of your choice (e.g., Google Gemini's free tier). 
# You are free to choose any domain (e.g., weather, stocks, games), but your server must implement at least two distinct tools using public APIs (ideally free - no need to spend money on this) or local data. 
# Submit your Python source code and a screenshot demonstrating the system successfully answering a prompt by executing your custom tools. [30 points]

# Cell-15

mcp_code = r"""
from fastapi import FastAPI, HTTPException
import random, requests

app = FastAPI()

@app.get("/tools")
def list_tools():
    return {"tools":[
        {"name":"randomNumber","description":"Return random integer between min and max","inputs":{"min":"int","max":"int"}},
        {"name":"weatherByCity","description":"Return current weather for a city using Open-Meteo","inputs":{"city":"string"}}
    ]}

@app.get("/tool/randomNumber")
def random_number(min: int = 0, max: int = 10):
    if min > max:
        raise HTTPException(status_code=400, detail="min must be <= max")
    return {"tool":"randomNumber","min":min,"max":max,"result":random.randint(min,max)}

@app.get("/tool/weatherByCity")
def weather_by_city(city: str):
    city_map = {"london": (51.5074, -0.1278), "new york": (40.7128, -74.0060), "san francisco": (37.7749, -122.4194)}
    key = city.lower()
    if key not in city_map:
        raise HTTPException(status_code=404, detail="City not in demo mapping")
    lat, lon = city_map[key]
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Weather API error")
    return {"tool":"weatherByCity","city":city,"result":resp.json().get("current_weather")}
"""
Path("mcp_server.py").write_text(mcp_code)
print("Wrote mcp_server.py")

# Cell-16

import threading, subprocess, sys, time
import nest_asyncio

nest_asyncio.apply()

BASE_URL = "http://127.0.0.1:8000"

def run_server():
    cmd = [
        sys.executable,
        "-m", "uvicorn",
        "mcp_server:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--log-level", "info",
    ]
    subprocess.run(cmd)

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

time.sleep(2)
print("Server started at:", BASE_URL)

# Cell-17

print("Tools:", requests.get(BASE_URL + "/tools").json())
print("Random number:", requests.get(BASE_URL + "/tool/randomNumber?min=5&max=15").json())
print("Weather London:", requests.get(BASE_URL + "/tool/weatherByCity?city=London").json())

# Cell-18

tool_prompt = f"""
You are an agent that can call tools via HTTP.

Tools:
1) randomNumber(min:int, max:int)
2) weatherByCity(city:string)

Return STRICT JSON with a key "calls" whose value is a list of calls.
Each call is {{"tool":"toolName","args":{{...}}}}.

Example:
{{
  "calls": [
    {{"tool":"weatherByCity","args":{{"city":"London"}}}},
    {{"tool":"randomNumber","args":{{"min":1,"max":10}}}}
  ]
}}

Task: Produce calls to get the current weather in London, then a random number between 1 and 10.
Only output the JSON.
"""

llm_text = call_gemini(tool_prompt, max_output_tokens=300, temperature=0.0)
print("Gemini raw output:\n", llm_text)

m = re.search(r"\{.*\}", llm_text, flags=re.S)
if not m:
    raise RuntimeError("Could not find JSON in Gemini output; inspect raw output.")

payload = json.loads(m.group(0))
print("Parsed JSON payload:", payload)

results = []
for call in payload.get("calls", []):
    tool = call["tool"]
    args = call.get("args", {})

    if tool == "weatherByCity":
        r = requests.get(BASE_URL + f"/tool/weatherByCity?city={args['city']}").json()
    elif tool == "randomNumber":
        r = requests.get(BASE_URL + f"/tool/randomNumber?min={args['min']}&max={args['max']}").json()
    else:
        r = {"error": "unknown tool"}

    results.append({"tool": tool, "args": args, "result": r})

print("Tool execution results:\n", json.dumps(results, indent=2))
