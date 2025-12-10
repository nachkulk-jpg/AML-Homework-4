#Cell-1

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

#Cell-2

DATA_ROOT = Path("/content/drive/MyDrive/image_files/v0")
print("Dataset root:", DATA_ROOT)

conditions = sorted([p.name for p in DATA_ROOT.iterdir() if p.is_dir()])
print("Detected conditions:", conditions)

#Cell-3

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
