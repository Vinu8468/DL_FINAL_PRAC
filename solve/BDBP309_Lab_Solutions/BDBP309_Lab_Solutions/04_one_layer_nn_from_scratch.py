"""Lab 4: One-layer neural network trained from scratch on the handout dataset."""
import numpy as np
import matplotlib.pyplot as plt

# ---------- Dataset copied from Lab 4 table: x1, x2, x3, y
vin_data = np.array([
    [0, 0, 1, 0],
    [1, 1, 1, 1],
    [1, 0, 1, 1],
    [0, 1, 1, 0]
], dtype=float)

vin_x = vin_data[:, :-1]
vin_y = vin_data[:, -1:]

np.random.seed(1)
vin_weights = np.random.randn(3, 1) * 0.1
vin_bias = np.zeros((1, 1))
vin_learning_rate = 0.1
vin_losses = []

# ---------- Sigmoid model and binary cross-entropy
for vin_epoch in range(1000):
    vin_z = np.dot(vin_x, vin_weights) + vin_bias
    vin_prediction = 1.0 / (1.0 + np.exp(-vin_z))
    vin_epsilon = 1e-8
    vin_loss = -np.mean(
        vin_y * np.log(vin_prediction + vin_epsilon)
        + (1 - vin_y) * np.log(1 - vin_prediction + vin_epsilon)
    )
    vin_losses.append(vin_loss)

    # For sigmoid + binary cross-entropy, derivative with respect to z is prediction - target
    vin_dz = (vin_prediction - vin_y) / len(vin_x)
    vin_dw = np.dot(vin_x.T, vin_dz)
    vin_db = np.sum(vin_dz, axis=0, keepdims=True)
    vin_weights = vin_weights - vin_learning_rate * vin_dw
    vin_bias = vin_bias - vin_learning_rate * vin_db

print("Final predictions:")
print((vin_prediction >= 0.5).astype(int))
print("Targets:")
print(vin_y.astype(int))
print("Final loss:", vin_losses[-1])

plt.plot(vin_losses)
plt.title("Training loss - one-layer network")
plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy")
plt.grid(True)
plt.show()
