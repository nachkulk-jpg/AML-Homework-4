# Cell-1

import os
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import pickle
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

# Cell-2

DATA_ROOT = Path("/content/drive/MyDrive/image_files/v0")
print("Dataset root:", DATA_ROOT)

conditions = sorted([p.name for p in DATA_ROOT.iterdir() if p.is_dir()])
print("Detected conditions:", conditions)

# Cell-3

CATEGORIES = ["airplane", "car", "chair", "cup", "dog", "donkey", "duck", "hat"]

for cond in conditions:
    folder = DATA_ROOT / cond
    imgs = sorted([f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
    print(f"{cond:12s} - {len(imgs):4d} images - sample: {imgs[0].name if imgs else 'NONE'}")


# Question-1

# 1. From Huggingface (https://huggingface.co/ Links to an external site.) select and download a pre-trained CLIP model (you can use your own computer, Colab, Kaggle... to store the model). 
    # Describe the model you downloaded - what is its architecture (e.g. CNN/ViT), number of layers, parameters per layer - breakdown the parameters and explain what they are doing (e.g., are they parts of K, Q and V matrices, bias, feature maps, dense layer...). [15 points]

# Cell-4

model_name = "openai/clip-vit-base-patch32"
print("Loading model:", model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(device)
model.eval()
print("Model loaded.")

def count_params_in_module(mod):
    return sum(p.numel() for p in mod.parameters())

vision_params = count_params_in_module(model.vision_model)
text_params = count_params_in_module(model.text_model)
proj_params = 0
if hasattr(model, "visual_projection"):
    proj_params += count_params_in_module(model.visual_projection) if model.visual_projection is not None else 0
if hasattr(model, "text_projection"):
    proj_params += count_params_in_module(model.text_projection) if model.text_projection is not None else 0

total_params = count_params_in_module(model)

print(f"Total parameters: {total_params:,}")
print(f" - Vision encoder (model.vision_model): {vision_params:,}")
print(f" - Text encoder   (model.text_model):  {text_params:,}")
print(f" - Projection heads (text/visual):    {proj_params:,}")

print("\nVision module type:", type(model.vision_model))
print("Text module type:", type(model.text_model))

vision_config = getattr(model.vision_model, "config", None)
text_config = getattr(model.text_model, "config", None)
if vision_config is not None:
    vc = vision_config
    print("\nVision config (ViT):")
    for k in ["hidden_size", "num_hidden_layers", "num_attention_heads", "patch_size", "image_size", "intermediate_size"]:
        if hasattr(vc, k):
            print(f"  {k}: {getattr(vc,k)}")
if text_config is not None:
    tc = text_config
    print("\nText config (Transformer):")
    for k in ["hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size", "vocab_size", "max_position_embeddings"]:
        if hasattr(tc, k):
            print(f"  {k}: {getattr(tc,k)}")


# Question-2

# 2. The dataset contains images from the following eight categories: airplane, car, chair, cup, dog, donkey, duck and hat. 
# Each category contains images in five different conditions: realistic, geons, silhouettes, blured and features. 
# Evaluate the model for each condition separately. 
# For each image in the dataset, feed the image into the model together with a text label of a particular category (for each image, evaluate labels of all eight categories). 
# If the model outputs highest correlation for the correct label, consider that as correct classification and otherwise as incorrect classification. 
# Create a confusion matrix and quantify model accuracy for each of the five conditions. [20 points]

# Cell-5

labels = CATEGORIES[:]
prompts = labels
print("Labels/prompts:", prompts)

# Cell-6

def infer_label_from_filename(fname, labels):
    n = fname.lower()
    for i, lab in enumerate(labels):
        if lab in n:
            return i
    for i, lab in enumerate(labels):
        if n.startswith(lab[:3]):
            return i
    return -1

sample_folder = DATA_ROOT / conditions[0]
sample_files = sorted([f.name for f in sample_folder.iterdir()][:10])
print("Sample filenames:", sample_files)
for s in sample_files:
    print(s, "->", infer_label_from_filename(s, labels))

# Cell-7

from typing import List, Tuple

def predict_batch_images(image_paths: List[Path], prompts: List[str], batch_size:int=16) -> np.ndarray:
    logits_all = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image
            logits_all.append(logits.cpu().numpy())
    return np.vstack(logits_all)

# Cell-8

results = {}

for cond in conditions:
    folder = DATA_ROOT / cond
    image_paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
    if len(image_paths) == 0:
        print("No images in", cond)
        continue

    logits_matrix = predict_batch_images(image_paths, prompts, batch_size=32)
    probs = (np.exp(logits_matrix) / np.exp(logits_matrix).sum(axis=1, keepdims=True))

    cond_results = []
    for idx, pth in enumerate(image_paths):
        true_idx = infer_label_from_filename(pth.name, labels)
        pred_idx = int(np.argmax(probs[idx]))
        cond_results.append((true_idx, pred_idx, probs[idx], str(pth)))
    results[cond] = cond_results
    print(f"Evaluated condition '{cond}' : {len(cond_results)} images")

# Cell-9

stats = {}
for cond, res_list in results.items():
    y_true = [t for (t,p,prob,path) in res_list if t >= 0]
    y_pred = [p for (t,p,prob,path) in res_list if t >= 0]
    if len(y_true) == 0:
        print("No labeled images (couldn't infer) for", cond)
        continue
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    acc = accuracy_score(y_true, y_pred)
    stats[cond] = {"cm": cm, "accuracy": acc, "n": len(y_true)}
    print(f"{cond:12s} N={len(y_true):3d}  Accuracy={acc*100:5.2f}%")

print("\n")
summary = pd.DataFrame([
    {"condition": cond, "n": stats[cond]["n"], "accuracy": stats[cond]["accuracy"]}
    for cond in stats
]).sort_values("condition").reset_index(drop=True)
summary["accuracy_pct"] = summary["accuracy"] * 100
display(summary)

# Cell-10

def plot_cm(cm, labels, title):
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()

for cond in stats:
    print(f"\n\nCondition: {cond}  (N={stats[cond]['n']})\n")
    plot_cm(stats[cond]["cm"], labels, f"Confusion matrix — {cond}")

# Cell-11

for cond in stats:
    cm = stats[cond]["cm"]
    class_counts = cm.sum(axis=1)
    class_correct = np.diag(cm)
    per_class_acc = np.divide(class_correct, class_counts, out=np.zeros_like(class_correct, dtype=float), where=class_counts!=0)
    print(f"\n{cond} (N={stats[cond]['n']}):")
    for lab, acc in zip(labels, per_class_acc):
        print(f"  {lab:8s}: {acc*100:5.2f}%  (n={int(class_counts[labels.index(lab)])})")
