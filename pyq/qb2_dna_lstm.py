# B2: LSTM classifier for DNA sequences

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Generate 400 DNA sequences and labels
# --------------------------------------------------

def vin_generate_data():

    vin_seq = ["ACGTAGCTAGCT", "GCTAGCTAGGCA",
               "CGTACGTAGCTA", "TGCATGCATGCA"]

    vin_labels = [0, 1, 0, 1]

    vin_data = []
    vin_y = []

    # Repeat each sequence 100 times
    for i in range(100):
        for j in range(4):
            vin_data.append(vin_seq[j])
            vin_y.append(vin_labels[j])

    return vin_data, vin_y

vin_data, vin_y = vin_generate_data()

print("Total sequences:", len(vin_data))
print("Total labels:", len(vin_y))

# --------------------------------------------------
# 2. Convert DNA letters into numbers
# A=1, C=2, G=3, T=4
# --------------------------------------------------

vin_map = {'A':1, 'C':2, 'G':3, 'T':4}

vin_x = []

for vin_s in vin_data:
    vin_temp = []

    for vin_char in vin_s:
        vin_temp.append(vin_map[vin_char])

    vin_x.append(vin_temp)

vin_x = torch.tensor(vin_x, dtype=torch.long)
vin_y = torch.tensor(vin_y, dtype=torch.long)

# --------------------------------------------------
# 3. Split into 80% training and 20% testing
# --------------------------------------------------

vin_x_train, vin_x_test, vin_y_train, vin_y_test = train_test_split(
    vin_x, vin_y, test_size=0.2, random_state=42
)

vin_train = TensorDataset(vin_x_train, vin_y_train)
vin_test = TensorDataset(vin_x_test, vin_y_test)

vin_train_loader = DataLoader(vin_train, batch_size=32, shuffle=True)
vin_test_loader = DataLoader(vin_test, batch_size=32, shuffle=False)

# --------------------------------------------------
# 4. Define LSTM model
# --------------------------------------------------

class VinLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(5, 8)
        self.lstm = nn.LSTM(8, 16, batch_first=True)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):

        x = self.embed(x)
        vin_out, (vin_h, vin_c) = self.lstm(x)

        x = vin_h[-1]
        x = self.fc(x)

        return x

vin_model = VinLSTM()

# --------------------------------------------------
# 5. Train model for 50 epochs
# --------------------------------------------------

vin_loss = nn.CrossEntropyLoss()
vin_optimizer = optim.Adam(vin_model.parameters(), lr=0.001)

for epoch in range(50):

    vin_total_loss = 0

    for vin_seq, vin_labels in vin_train_loader:

        vin_outputs = vin_model(vin_seq)
        vin_l = vin_loss(vin_outputs, vin_labels)

        vin_optimizer.zero_grad()
        vin_l.backward()
        vin_optimizer.step()

        vin_total_loss += vin_l.item()

    # Print loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print("Epoch:", epoch+1, "Loss:", vin_total_loss)

# --------------------------------------------------
# 6. Test model
# --------------------------------------------------

vin_correct = 0
vin_total = 0

with torch.no_grad():

    for vin_seq, vin_labels in vin_test_loader:

        vin_outputs = vin_model(vin_seq)
        vin_pred = torch.argmax(vin_outputs, dim=1)

        vin_correct += (vin_pred == vin_labels).sum().item()
        vin_total += len(vin_labels)

print("Test accuracy:", 100 * vin_correct / vin_total, "%")
