"""Lab 13: Predict a binary gene-accessibility label from DNA sequence using a 1D CNN.

Optional CSV: accessibility.csv with columns sequence,label (label is 0 or 1).
Without the file, a synthetic motif-based example is used for demonstration.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

vin_alphabet = "ACGT"

def vin_one_hot(vin_sequence, vin_max_length=100):
    vin_matrix = np.zeros((4, vin_max_length), dtype=np.float32)
    vin_sequence = vin_sequence.upper()[:vin_max_length]
    for vin_position in range(len(vin_sequence)):
        vin_base = vin_sequence[vin_position]
        if vin_base in vin_alphabet:
            vin_matrix[vin_alphabet.index(vin_base), vin_position] = 1.0
    return vin_matrix

def vin_integer_encode(vin_sequence, vin_max_length=100):
    vin_base_to_integer = {"A": 1, "C": 2, "G": 3, "T": 4}
    vin_encoded = np.zeros(vin_max_length, dtype=np.int64)
    vin_sequence = vin_sequence.upper()[:vin_max_length]
    for vin_position in range(len(vin_sequence)):
        vin_encoded[vin_position] = vin_base_to_integer.get(vin_sequence[vin_position], 0)
    return vin_encoded

def vin_kmer_encode(vin_sequence, vin_k=3):
    vin_counts = {}
    vin_sequence = vin_sequence.upper()
    for vin_position in range(len(vin_sequence) - vin_k + 1):
        vin_kmer = vin_sequence[vin_position:vin_position + vin_k]
        vin_counts[vin_kmer] = vin_counts.get(vin_kmer, 0) + 1
    return vin_counts

# ---------- Compare the three sequence representations
vin_example_sequence = "ATGCGTACGT"
print("One-hot shape:", vin_one_hot(vin_example_sequence).shape)
print("Integer encoding:", vin_integer_encode(vin_example_sequence))
print("3-mer counts:", vin_kmer_encode(vin_example_sequence, 3))
# One-hot preserves base identity and position; integer encoding is compact but
# its numeric values do not imply that T is biologically 'larger' than A.
# k-mer counts capture short motifs but discard their exact locations.

if os.path.exists("accessibility.csv"):
    vin_df = pd.read_csv("accessibility.csv")
    vin_sequences = vin_df["sequence"].astype(str).tolist()
    vin_labels = vin_df["label"].astype(int).values
    print("Loaded accessibility.csv")
else:
    np.random.seed(11)
    vin_sequences = []
    vin_label_list = []
    for vin_i in range(300):
        vin_seq = "".join(np.random.choice(list(vin_alphabet), 100))
        vin_label = 1 if "CGCG" in vin_seq or "GCGC" in vin_seq else 0
        vin_sequences.append(vin_seq)
        vin_label_list.append(vin_label)
    vin_labels = np.array(vin_label_list)
    print("Using synthetic motif-based sequences; replace with real accessibility.csv for biological analysis.")

vin_x = np.array([vin_one_hot(vin_seq) for vin_seq in vin_sequences])
vin_x_train, vin_x_test, vin_y_train, vin_y_test = train_test_split(
    vin_x, vin_labels, test_size=0.2, random_state=42, stratify=vin_labels
)
vin_train_loader = DataLoader(
    TensorDataset(torch.tensor(vin_x_train), torch.tensor(vin_y_train, dtype=torch.long)),
    batch_size=32, shuffle=True
)

class VinSequenceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten(),
            nn.Linear(64, 2)
        )
    def forward(self, vin_input):
        return self.network(vin_input)

vin_model = VinSequenceCNN()
vin_optimizer = torch.optim.Adam(vin_model.parameters(), lr=0.001)
vin_loss_function = nn.CrossEntropyLoss()
for vin_epoch in range(15):
    vin_model.train()
    for vin_batch_x, vin_batch_y in vin_train_loader:
        vin_output = vin_model(vin_batch_x)
        vin_loss = vin_loss_function(vin_output, vin_batch_y)
        vin_optimizer.zero_grad()
        vin_loss.backward()
        vin_optimizer.step()

vin_model.eval()
with torch.no_grad():
    vin_test_x = torch.tensor(vin_x_test)
    vin_test_y = torch.tensor(vin_y_test)
    vin_prediction = torch.argmax(vin_model(vin_test_x), dim=1)
    print("Test accuracy:", (vin_prediction == vin_test_y).float().mean().item())
torch.save(vin_model.state_dict(), "13_gene_accessibility_cnn.pth")
