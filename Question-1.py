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
