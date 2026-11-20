import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Generate DNA dataset
def vin_generate_data():
    vin_sequences = [
        "CAGTAGCTAGCT",
        "CGTAGCTGAGCA",
        "CGTACGTAGCTA",
        "TGCATGCATGCA",
        "GCTACGTAGGCA",
        "GCTACGTAGCTA"
    ]

    vin_labels = [1, 0, 1, 0, 1, 0]

    vin_X = []
    vin_y = []

    for i in range(len(vin_sequences)):
        for j in range(100):
            vin_X.append(vin_sequences[i])
            vin_y.append(vin_labels[i])

    return vin_X, vin_y

# Encode DNA characters
def vin_encode(vin_sequence):
    vin_mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    vin_encoded = []

    for vin_char in vin_sequence:
        vin_encoded.append(vin_mapping[vin_char])

    return vin_encoded

# Generate dataset
vin_X, vin_y = vin_generate_data()

vin_X = [vin_encode(seq) for seq in vin_X]

vin_X = torch.tensor(vin_X, dtype=torch.long)
vin_y = torch.tensor(vin_y, dtype=torch.float32)

# Split dataset
vin_X_train, vin_X_test, vin_y_train, vin_y_test = train_test_split(
    vin_X, vin_y, test_size=0.2, random_state=42
)

# DataLoader
vin_train_dataset = TensorDataset(vin_X_train, vin_y_train)
vin_train_loader = DataLoader(vin_train_dataset, batch_size=32, shuffle=True)

# GRU model
class vin_GRU(nn.Module):
    def __init__(self):
        super(vin_GRU, self).__init__()

        self.embedding = nn.Embedding(4, 8)
        self.gru = nn.GRU(8, 16, batch_first=True)
        self.fc = nn.Linear(16, 1)

    def forward(self, vin_x):
        vin_x = self.embedding(vin_x)
        vin_x, vin_hidden = self.gru(vin_x)
        vin_x = vin_x[:, -1, :]
        vin_x = self.fc(vin_x)
        vin_x = torch.sigmoid(vin_x)

        return vin_x

# Initialize model
vin_model = vin_GRU()
vin_criterion = nn.BCELoss()
vin_optimizer = torch.optim.Adam(vin_model.parameters(), lr=0.001)

# Training
for vin_epoch in range(50):

    vin_total_loss = 0

    for vin_batch_X, vin_batch_y in vin_train_loader:

        vin_output = vin_model(vin_batch_X)
        vin_batch_y = vin_batch_y.reshape(-1, 1)

        vin_loss = vin_criterion(vin_output, vin_batch_y)

        vin_optimizer.zero_grad()
        vin_loss.backward()
        vin_optimizer.step()

        vin_total_loss += vin_loss.item()

    if (vin_epoch + 1) % 10 == 0:
        print("Epoch:", vin_epoch + 1,
              "Loss:", vin_total_loss)

# Testing
with torch.no_grad():

    vin_output = vin_model(vin_X_test)
    vin_predictions = (vin_output >= 0.5).float()

    vin_accuracy = (vin_predictions == vin_y_test.reshape(-1, 1)).float().mean()

    print("Test Accuracy:", vin_accuracy.item() * 100, "%")
