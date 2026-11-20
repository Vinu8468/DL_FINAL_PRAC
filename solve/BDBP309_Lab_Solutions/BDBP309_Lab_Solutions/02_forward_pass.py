"""Lab 2: Forward pass for the single-neuron and multilayer networks in the handout."""
import numpy as np

np.random.seed(10)

def vin_relu(vin_z):
    return np.maximum(0, vin_z)

# ---------- Network 1: four inputs -> one output neuron
vin_x = np.random.randn(4)
vin_w = np.random.randn(1, 4)
vin_b = np.random.randn(1)
vin_z = np.dot(vin_w, vin_x) + vin_b
vin_a = vin_relu(vin_z)
print("Network 1")
print("Input:", vin_x)
print("z:", vin_z)
print("activation:", vin_a)

# ---------- Network 2: 4 -> 3 -> 2 -> 1
vin_x = np.random.randn(4)
vin_w1 = np.random.randn(3, 4)
vin_b1 = np.random.randn(3)
vin_w2 = np.random.randn(2, 3)
vin_b2 = np.random.randn(2)
vin_w3 = np.random.randn(1, 2)
vin_b3 = np.random.randn(1)

# Vectorized forward pass: no neuron-wise loops
vin_z1 = np.dot(vin_w1, vin_x) + vin_b1
vin_a1 = vin_relu(vin_z1)
vin_z2 = np.dot(vin_w2, vin_a1) + vin_b2
vin_a2 = vin_relu(vin_z2)
vin_z3 = np.dot(vin_w3, vin_a2) + vin_b3
vin_y_hat = vin_z3   # linear output for this demonstration

print("\nNetwork 2")
print("Input:", vin_x)
print("Layer 1 activations:", vin_a1)
print("Layer 2 activations:", vin_a2)
print("Output prediction:", vin_y_hat)

# ---------- Example mean squared error if a target is available
vin_y = np.array([1.0])
vin_loss = np.mean((vin_y_hat - vin_y) ** 2)
print("MSE loss:", vin_loss)
