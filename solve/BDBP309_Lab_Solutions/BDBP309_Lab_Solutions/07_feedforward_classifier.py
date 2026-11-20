"""Lab 7: End-to-end PyTorch feedforward classifier. Uses sklearn breast-cancer data."""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------- Data loading options (choose one)
# CSV: vin_df = pd.read_csv("data.csv"); vin_x = vin_df.drop(columns=["label"]).values; vin_y = vin_df["label"].values
# NumPy: vin_arr = np.load("data.npy"); vin_x = vin_arr[:, :-1]; vin_y = vin_arr[:, -1]
# Excel: vin_df = pd.read_excel("data.xlsx")   # requires openpyxl
# This practical proceeds with a small built-in tabular dataset.
vin_dataset = load_breast_cancer()
vin_x = vin_dataset.data
vin_y = vin_dataset.target
vin_x_train, vin_x_test, vin_y_train, vin_y_test = train_test_split(
    vin_x, vin_y, test_size=0.2, random_state=42, stratify=vin_y
)
vin_scaler = StandardScaler()
vin_x_train = vin_scaler.fit_transform(vin_x_train)
vin_x_test = vin_scaler.transform(vin_x_test)

vin_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vin_x_train = torch.tensor(vin_x_train, dtype=torch.float32)
vin_x_test = torch.tensor(vin_x_test, dtype=torch.float32)
vin_y_train = torch.tensor(vin_y_train, dtype=torch.long)
vin_y_test = torch.tensor(vin_y_test, dtype=torch.long)
vin_train_loader = DataLoader(TensorDataset(vin_x_train, vin_y_train), batch_size=32, shuffle=True)

class VinClassifier(nn.Module):
    def __init__(self, vin_input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(vin_input_size, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, vin_input):
        return self.network(vin_input)

vin_model = VinClassifier(vin_x_train.shape[1]).to(vin_device)
vin_optimizer = torch.optim.Adam(vin_model.parameters(), lr=0.001)
vin_loss_function = nn.CrossEntropyLoss()

for vin_epoch in range(30):
    vin_model.train()
    vin_total_loss = 0.0
    for vin_batch_x, vin_batch_y in vin_train_loader:
        vin_batch_x = vin_batch_x.to(vin_device)
        vin_batch_y = vin_batch_y.to(vin_device)
        vin_output = vin_model(vin_batch_x)
        vin_loss = vin_loss_function(vin_output, vin_batch_y)
        vin_optimizer.zero_grad()
        vin_loss.backward()
        vin_optimizer.step()
        vin_total_loss = vin_total_loss + vin_loss.item()
    print("Epoch", vin_epoch + 1, "loss", vin_total_loss / len(vin_train_loader))

vin_model.eval()
with torch.no_grad():
    vin_test_output = vin_model(vin_x_test.to(vin_device))
    vin_prediction = torch.argmax(vin_test_output, dim=1).cpu()
    vin_accuracy = (vin_prediction == vin_y_test).float().mean().item()
print("Test accuracy:", round(vin_accuracy, 4))
torch.save(vin_model.state_dict(), "07_feedforward_classifier.pth")
