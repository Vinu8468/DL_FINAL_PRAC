"""Reference sheet: load common data formats and separate X (features) from y (labels).

Run this file after adding your own files. Most examples are commented so the script
runs without requiring every format to be present. Install pandas, openpyxl, pillow,
torch and torchvision for all examples.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from PIL import Image

# --------------------------------------------------
# 1. CSV with pandas
# --------------------------------------------------
# vin_df = pd.read_csv("data.csv")
# vin_x = vin_df.drop(columns=["label"]).values
# vin_y = vin_df["label"].values

# --------------------------------------------------
# 2. Excel (.xlsx) - requires openpyxl
# --------------------------------------------------
# vin_df = pd.read_excel("data.xlsx", sheet_name=0)
# vin_x = vin_df.drop(columns=["target"]).values
# vin_y = vin_df["target"].values

# --------------------------------------------------
# 3. JSON / JSON Lines
# --------------------------------------------------
# vin_df = pd.read_json("data.json")
# For JSON Lines: vin_df = pd.read_json("data.jsonl", lines=True)
# vin_x = vin_df.drop(columns=["label"]).values
# vin_y = vin_df["label"].values

# --------------------------------------------------
# 4. Parquet - requires pyarrow or fastparquet
# --------------------------------------------------
# vin_df = pd.read_parquet("data.parquet")
# vin_x = vin_df.drop(columns=["label"]).values
# vin_y = vin_df["label"].values

# --------------------------------------------------
# 5. NumPy .npy (one array)
# --------------------------------------------------
# vin_array = np.load("data.npy")
# If last column contains the label:
# vin_x = vin_array[:, :-1]
# vin_y = vin_array[:, -1]

# --------------------------------------------------
# 6. NumPy .npz (multiple named arrays)
# --------------------------------------------------
# vin_data = np.load("data.npz")
# print(vin_data.files)             # see available array names
# vin_x = vin_data["X"]
# vin_y = vin_data["y"]

# --------------------------------------------------
# 7. Text files / corpus
# --------------------------------------------------
# with open("corpus.txt", "r", encoding="utf-8") as vin_file:
#     vin_lines = vin_file.readlines()
# vin_texts = [vin_line.strip() for vin_line in vin_lines]

# --------------------------------------------------
# 8. FASTA sequence file
# --------------------------------------------------
def vin_read_fasta(vin_path):
    vin_records = []
    vin_header = None
    vin_sequence = ""
    with open(vin_path, "r", encoding="utf-8") as vin_file:
        for vin_line in vin_file:
            vin_line = vin_line.strip()
            if vin_line.startswith(">"):
                if vin_header is not None:
                    vin_records.append((vin_header, vin_sequence))
                vin_header = vin_line[1:]
                vin_sequence = ""
            else:
                vin_sequence = vin_sequence + vin_line
    if vin_header is not None:
        vin_records.append((vin_header, vin_sequence))
    return vin_records

# Example: vin_fasta_records = vin_read_fasta("sequences.fasta")
# vin_ids = [vin_record[0] for vin_record in vin_fasta_records]
# vin_sequences = [vin_record[1] for vin_record in vin_fasta_records]

# --------------------------------------------------
# 9. One image using Pillow
# --------------------------------------------------
# vin_image = Image.open("image.jpg").convert("RGB")

# --------------------------------------------------
# 10. Image folder using torchvision ImageFolder
# --------------------------------------------------
# from torchvision import datasets, transforms
# vin_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
# vin_dataset = datasets.ImageFolder("image_dataset/train", transform=vin_transform)
# vin_loader = DataLoader(vin_dataset, batch_size=32, shuffle=True)
# Class folder structure: image_dataset/train/class_0/*.jpg and class_1/*.jpg

# --------------------------------------------------
# 11. Convert NumPy arrays to PyTorch tensors and DataLoader
# --------------------------------------------------
def vin_make_loader(vin_x, vin_y, vin_batch_size=32, vin_shuffle=True, vin_target_dtype=torch.long):
    vin_x_tensor = torch.tensor(np.asarray(vin_x), dtype=torch.float32)
    # Classification labels use torch.long; regression targets usually use torch.float32.
    vin_y_tensor = torch.tensor(np.asarray(vin_y), dtype=vin_target_dtype)
    vin_dataset = TensorDataset(vin_x_tensor, vin_y_tensor)
    return DataLoader(vin_dataset, batch_size=vin_batch_size, shuffle=vin_shuffle)

# --------------------------------------------------
# 12. Custom PyTorch Dataset template
# --------------------------------------------------
class VinTabularDataset(Dataset):
    def __init__(self, vin_x, vin_y):
        self.vin_x = torch.tensor(np.asarray(vin_x), dtype=torch.float32)
        self.vin_y = torch.tensor(np.asarray(vin_y), dtype=torch.long)

    def __len__(self):
        return len(self.vin_x)

    def __getitem__(self, vin_index):
        return self.vin_x[vin_index], self.vin_y[vin_index]

# --------------------------------------------------
# 13. A tiny real runnable example using a local CSV, if present
# --------------------------------------------------
if os.path.exists("sample_data/tabular_demo.csv"):
    vin_demo_df = pd.read_csv("sample_data/tabular_demo.csv")
    vin_x = vin_demo_df.select_dtypes(include=[np.number]).drop(columns=["label"]).values
    vin_y = vin_demo_df["label"].values
    vin_loader = vin_make_loader(vin_x, vin_y, vin_batch_size=2)
    for vin_batch_x, vin_batch_y in vin_loader:
        print("Loaded batch X shape:", vin_batch_x.shape)
        print("Loaded batch y:", vin_batch_y)
        break
else:
    print("Data loading reference ready. Uncomment a loader example and set your file path.")
