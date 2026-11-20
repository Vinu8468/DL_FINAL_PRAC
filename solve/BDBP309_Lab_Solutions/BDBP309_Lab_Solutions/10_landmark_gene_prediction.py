"""Lab 10: Predict a target gene from landmark gene expression values."""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ---------- Expected real-data CSV format
# gene_expression.csv: one row per sample; columns include landmark genes and target_gene.
# For example: sample_id, TP53, EGFR, MYC, ... , target_gene
# Load with: vin_df = pd.read_csv("gene_expression.csv")
# Split: vin_x = vin_df[["TP53", "EGFR", "MYC"]].values; vin_y = vin_df["target_gene"].values
# Excel alternative: pd.read_excel("gene_expression.xlsx")

if os.path.exists("gene_expression.csv"):
    vin_df = pd.read_csv("gene_expression.csv")
    vin_target_column = "target_gene"
    vin_x = vin_df.drop(columns=[vin_target_column]).select_dtypes(include=[np.number]).values
    vin_y = vin_df[vin_target_column].values.astype(float)
    print("Loaded gene_expression.csv")
else:
    # Clearly marked synthetic demonstration data, not a biological result.
    np.random.seed(10)
    vin_x = np.random.randn(300, 20)
    vin_y = 1.5 * vin_x[:, 0] - 0.8 * vin_x[:, 2] + 0.4 * vin_x[:, 5] + np.random.randn(300) * 0.2
    print("gene_expression.csv not found; using synthetic demonstration data.")

vin_x_train, vin_x_test, vin_y_train, vin_y_test = train_test_split(
    vin_x, vin_y, test_size=0.2, random_state=42
)
vin_scaler = StandardScaler()
vin_x_train = vin_scaler.fit_transform(vin_x_train)
vin_x_test = vin_scaler.transform(vin_x_test)

vin_x_train = torch.tensor(vin_x_train, dtype=torch.float32)
vin_x_test = torch.tensor(vin_x_test, dtype=torch.float32)
vin_y_train = torch.tensor(vin_y_train.reshape(-1, 1), dtype=torch.float32)
vin_y_test = torch.tensor(vin_y_test.reshape(-1, 1), dtype=torch.float32)
vin_loader = DataLoader(TensorDataset(vin_x_train, vin_y_train), batch_size=32, shuffle=True)

class VinGeneRegressor(nn.Module):
    def __init__(self, vin_num_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(vin_num_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, vin_input):
        return self.network(vin_input)

vin_model = VinGeneRegressor(vin_x_train.shape[1])
vin_optimizer = torch.optim.Adam(vin_model.parameters(), lr=0.001)
vin_loss_function = nn.MSELoss()
for vin_epoch in range(100):
    vin_model.train()
    for vin_batch_x, vin_batch_y in vin_loader:
        vin_prediction = vin_model(vin_batch_x)
        vin_loss = vin_loss_function(vin_prediction, vin_batch_y)
        vin_optimizer.zero_grad()
        vin_loss.backward()
        vin_optimizer.step()

vin_model.eval()
with torch.no_grad():
    vin_test_prediction = vin_model(vin_x_test).numpy().ravel()
print("Test MSE:", mean_squared_error(vin_y_test.numpy().ravel(), vin_test_prediction))
print("Test R2:", r2_score(vin_y_test.numpy().ravel(), vin_test_prediction))
torch.save(vin_model.state_dict(), "10_landmark_gene_model.pth")
