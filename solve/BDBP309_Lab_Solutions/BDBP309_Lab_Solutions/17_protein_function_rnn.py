"""Lab 17: Predict protein function labels from amino-acid sequences using a GRU.

Optional protein_functions.csv columns: sequence,functions.
Use semicolon-separated function labels, e.g. "binding;enzyme".
A FASTA parser example is included below; labels should be joined from annotation data.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

vin_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
vin_amino_to_index = {vin_amino_acids[vin_i]: vin_i + 1 for vin_i in range(len(vin_amino_acids))}
vin_amino_to_index["X"] = 0
vin_max_length = 100

def vin_encode_protein(vin_sequence):
    vin_result = np.zeros(vin_max_length, dtype=np.int64)
    vin_sequence = vin_sequence.upper()[:vin_max_length]
    for vin_position in range(len(vin_sequence)):
        vin_result[vin_position] = vin_amino_to_index.get(vin_sequence[vin_position], 0)
    return vin_result

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

if os.path.exists("protein_functions.csv"):
    vin_df = pd.read_csv("protein_functions.csv")
    vin_sequences = vin_df["sequence"].astype(str).tolist()
    vin_function_strings = vin_df["functions"].astype(str).tolist()
else:
    # Synthetic demonstration: two sequence motifs are associated with two example functions.
    np.random.seed(17)
    vin_sequences = []
    vin_function_strings = []
    for vin_i in range(100):
        vin_sequence = "".join(np.random.choice(list(vin_amino_acids), 80))
        if vin_i % 2 == 0:
            vin_sequence = "HIS" + vin_sequence
            vin_function_strings.append("binding")
        else:
            vin_sequence = "GLY" + vin_sequence
            vin_function_strings.append("enzyme")
        vin_sequences.append(vin_sequence)
    print("Using synthetic sequences; replace with labelled protein_functions.csv for biological analysis.")

vin_label_names = sorted(set(vin_label.strip() for vin_value in vin_function_strings for vin_label in vin_value.split(";")))
vin_label_to_index = {vin_label_names[vin_i]: vin_i for vin_i in range(len(vin_label_names))}
vin_y = np.zeros((len(vin_sequences), len(vin_label_names)), dtype=np.float32)
for vin_i in range(len(vin_sequences)):
    for vin_label in vin_function_strings[vin_i].split(";"):
        vin_y[vin_i, vin_label_to_index[vin_label.strip()]] = 1.0
vin_x = np.array([vin_encode_protein(vin_sequence) for vin_sequence in vin_sequences])
vin_x_train, vin_x_test, vin_y_train, vin_y_test = train_test_split(vin_x, vin_y, test_size=0.2, random_state=42)
vin_loader = DataLoader(
    TensorDataset(torch.tensor(vin_x_train), torch.tensor(vin_y_train)),
    batch_size=16, shuffle=True
)

class VinProteinGRU(nn.Module):
    def __init__(self, vin_num_functions):
        super().__init__()
        self.embedding = nn.Embedding(len(vin_amino_to_index) + 1, 16, padding_idx=0)
        self.gru = nn.GRU(16, 32, batch_first=True, bidirectional=True)
        self.output = nn.Linear(64, vin_num_functions)
    def forward(self, vin_input):
        vin_embedded = self.embedding(vin_input)
        vin_output, vin_hidden = self.gru(vin_embedded)
        vin_features = torch.cat((vin_hidden[-2], vin_hidden[-1]), dim=1)
        return self.output(vin_features)

vin_model = VinProteinGRU(len(vin_label_names))
vin_optimizer = torch.optim.Adam(vin_model.parameters(), lr=0.001)
vin_loss_function = nn.BCEWithLogitsLoss()
for vin_epoch in range(4):
    vin_model.train()
    vin_total_loss = 0.0
    for vin_batch_x, vin_batch_y in vin_loader:
        vin_logits = vin_model(vin_batch_x)
        vin_loss = vin_loss_function(vin_logits, vin_batch_y)
        vin_optimizer.zero_grad()
        vin_loss.backward()
        vin_optimizer.step()
        vin_total_loss = vin_total_loss + vin_loss.item()
    print("Epoch", vin_epoch + 1, "loss", vin_total_loss / len(vin_loader))

vin_model.eval()
with torch.no_grad():
    vin_test_logits = vin_model(torch.tensor(vin_x_test))
    vin_test_prediction = (torch.sigmoid(vin_test_logits).numpy() >= 0.5).astype(int)
    vin_micro_accuracy = np.mean(vin_test_prediction == vin_y_test)
print("Element-wise multilabel accuracy:", round(float(vin_micro_accuracy), 4))
print("Function labels:", vin_label_names)
torch.save(vin_model.state_dict(), "17_protein_function_gru.pth")
