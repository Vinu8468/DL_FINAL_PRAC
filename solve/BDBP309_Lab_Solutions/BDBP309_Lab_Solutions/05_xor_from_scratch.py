"""Lab 5: Two-layer neural network for XOR, from scratch using NumPy."""
import numpy as np
import matplotlib.pyplot as plt

# ---------- XOR truth table
vin_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
vin_y = np.array([[0], [1], [1], [0]], dtype=float)

np.random.seed(5)
vin_w1 = np.random.randn(2, 4) * 0.5
vin_b1 = np.zeros((1, 4))
vin_w2 = np.random.randn(4, 1) * 0.5
vin_b2 = np.zeros((1, 1))
vin_learning_rate = 0.5
vin_losses = []

def vin_sigmoid(vin_x):
    return 1.0 / (1.0 + np.exp(-vin_x))

# ---------- Training
for vin_epoch in range(10000):
    vin_z1 = np.dot(vin_x, vin_w1) + vin_b1
    vin_a1 = vin_sigmoid(vin_z1)
    vin_z2 = np.dot(vin_a1, vin_w2) + vin_b2
    vin_a2 = vin_sigmoid(vin_z2)

    vin_loss = np.mean((vin_a2 - vin_y) ** 2)
    vin_losses.append(vin_loss)

    vin_dz2 = (vin_a2 - vin_y) * vin_a2 * (1 - vin_a2)
    vin_dw2 = np.dot(vin_a1.T, vin_dz2) / len(vin_x)
    vin_db2 = np.mean(vin_dz2, axis=0, keepdims=True)

    vin_da1 = np.dot(vin_dz2, vin_w2.T)
    vin_dz1 = vin_da1 * vin_a1 * (1 - vin_a1)
    vin_dw1 = np.dot(vin_x.T, vin_dz1) / len(vin_x)
    vin_db1 = np.mean(vin_dz1, axis=0, keepdims=True)

    vin_w2 = vin_w2 - vin_learning_rate * vin_dw2
    vin_b2 = vin_b2 - vin_learning_rate * vin_db2
    vin_w1 = vin_w1 - vin_learning_rate * vin_dw1
    vin_b1 = vin_b1 - vin_learning_rate * vin_db1

# ---------- Predictions
vin_a1 = vin_sigmoid(np.dot(vin_x, vin_w1) + vin_b1)
vin_prediction = vin_sigmoid(np.dot(vin_a1, vin_w2) + vin_b2)
print("Input:\n", vin_x)
print("Predicted probabilities:\n", vin_prediction)
print("Predicted XOR:\n", (vin_prediction >= 0.5).astype(int))
print("True XOR:\n", vin_y.astype(int))

plt.plot(vin_losses)
plt.title("XOR training loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.show()
