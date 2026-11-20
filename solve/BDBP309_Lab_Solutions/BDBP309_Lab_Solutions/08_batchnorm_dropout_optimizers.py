"""Lab 8: BatchNorm, LayerNorm, dropout from scratch and optimizer comparison."""
import numpy as np
import torch
import torch.nn as nn

# ---------- Batch normalization from scratch (per feature)
def vin_batch_norm(vin_x, vin_gamma, vin_beta, vin_epsilon=1e-5):
    vin_mean = np.mean(vin_x, axis=0, keepdims=True)
    vin_variance = np.var(vin_x, axis=0, keepdims=True)
    vin_x_norm = (vin_x - vin_mean) / np.sqrt(vin_variance + vin_epsilon)
    return vin_gamma * vin_x_norm + vin_beta

# ---------- Layer normalization from scratch (per sample)
def vin_layer_norm(vin_x, vin_gamma, vin_beta, vin_epsilon=1e-5):
    vin_mean = np.mean(vin_x, axis=1, keepdims=True)
    vin_variance = np.var(vin_x, axis=1, keepdims=True)
    vin_x_norm = (vin_x - vin_mean) / np.sqrt(vin_variance + vin_epsilon)
    return vin_gamma * vin_x_norm + vin_beta

# ---------- Dropout from scratch
def vin_dropout(vin_x, vin_probability=0.5, vin_training=True):
    if not vin_training:
        return vin_x
    vin_mask = (np.random.rand(vin_x.shape[0], vin_x.shape[1]) > vin_probability).astype(float)
    return (vin_x * vin_mask) / (1.0 - vin_probability)

vin_x = np.random.randn(8, 4)
vin_gamma = np.ones((1, 4))
vin_beta = np.zeros((1, 4))
print("BatchNorm output:\n", vin_batch_norm(vin_x, vin_gamma, vin_beta))
print("LayerNorm output:\n", vin_layer_norm(vin_x, vin_gamma, vin_beta))
print("Dropout output:\n", vin_dropout(vin_x, 0.3))

# ---------- Compare PyTorch optimizers on the same tiny regression problem
vin_features = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
vin_targets = 3.0 * vin_features + 2.0
vin_optimizer_names = ["SGD", "Momentum", "Adagrad", "Adam"]

for vin_name in vin_optimizer_names:
    vin_model = nn.Linear(1, 1)
    if vin_name == "SGD":
        vin_optimizer = torch.optim.SGD(vin_model.parameters(), lr=0.05)
    elif vin_name == "Momentum":
        vin_optimizer = torch.optim.SGD(vin_model.parameters(), lr=0.05, momentum=0.9)
    elif vin_name == "Adagrad":
        vin_optimizer = torch.optim.Adagrad(vin_model.parameters(), lr=1.0)
    else:
        vin_optimizer = torch.optim.Adam(vin_model.parameters(), lr=0.05)
    vin_loss_function = nn.MSELoss()
    for vin_epoch in range(500):
        vin_prediction = vin_model(vin_features)
        vin_loss = vin_loss_function(vin_prediction, vin_targets)
        vin_optimizer.zero_grad()
        vin_loss.backward()
        vin_optimizer.step()
    print(vin_name, "final MSE:", round(vin_loss.item(), 6))
