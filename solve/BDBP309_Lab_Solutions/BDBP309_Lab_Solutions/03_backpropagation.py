"""Lab 3: Backpropagation for both networks from the handout using NumPy."""
import numpy as np

np.random.seed(2)

def vin_relu(vin_x):
    return np.maximum(0, vin_x)

def vin_relu_derivative(vin_x):
    return (vin_x > 0).astype(float)

# ---------- Network 1: four inputs -> one output neuron
vin_x_single = np.array([0.5, -1.0, 0.3, 0.8])
vin_y_single = np.array([1.0])
vin_w_single = np.random.randn(1, 4) * 0.2
vin_b_single = np.zeros(1)
vin_z_single = np.dot(vin_w_single, vin_x_single) + vin_b_single
vin_a_single = vin_relu(vin_z_single)
vin_loss_single = 0.5 * np.sum((vin_a_single - vin_y_single) ** 2)
vin_dz_single = (vin_a_single - vin_y_single) * vin_relu_derivative(vin_z_single)
vin_dw_single = np.outer(vin_dz_single, vin_x_single)
vin_db_single = vin_dz_single
print("Network 1 prediction:", vin_a_single)
print("Network 1 loss:", vin_loss_single)
print("Network 1 weight gradient:\n", vin_dw_single)
print("Network 1 bias gradient:", vin_db_single)

# ---------- Network 2: 4 inputs -> 3 neurons -> 2 neurons -> 1 output
vin_x = np.array([0.5, -1.0, 0.3, 0.8])
vin_y = np.array([1.0])
vin_w1 = np.random.randn(3, 4) * 0.2
vin_b1 = np.zeros(3)
vin_w2 = np.random.randn(2, 3) * 0.2
vin_b2 = np.zeros(2)
vin_w3 = np.random.randn(1, 2) * 0.2
vin_b3 = np.zeros(1)

# ---------- Forward pass
vin_z1 = np.dot(vin_w1, vin_x) + vin_b1
vin_a1 = vin_relu(vin_z1)
vin_z2 = np.dot(vin_w2, vin_a1) + vin_b2
vin_a2 = vin_relu(vin_z2)
vin_z3 = np.dot(vin_w3, vin_a2) + vin_b3
vin_y_hat = vin_z3
vin_loss = 0.5 * np.sum((vin_y_hat - vin_y) ** 2)

# ---------- Backward pass from output to first hidden layer
vin_dz3 = vin_y_hat - vin_y
vin_dw3 = np.outer(vin_dz3, vin_a2)
vin_db3 = vin_dz3
vin_da2 = np.dot(vin_w3.T, vin_dz3)
vin_dz2 = vin_da2 * vin_relu_derivative(vin_z2)
vin_dw2 = np.outer(vin_dz2, vin_a1)
vin_db2 = vin_dz2
vin_da1 = np.dot(vin_w2.T, vin_dz2)
vin_dz1 = vin_da1 * vin_relu_derivative(vin_z1)
vin_dw1 = np.outer(vin_dz1, vin_x)
vin_db1 = vin_dz1

print("\nNetwork 2 prediction:", vin_y_hat)
print("Network 2 loss:", vin_loss)
print("Layer 3 gradients dW3, db3:\n", vin_dw3, vin_db3)
print("Layer 2 gradients dW2, db2:\n", vin_dw2, vin_db2)
print("Layer 1 gradients dW1, db1:\n", vin_dw1, vin_db1)

# ---------- One SGD update for all layers
vin_learning_rate = 0.01
vin_w1 = vin_w1 - vin_learning_rate * vin_dw1
vin_b1 = vin_b1 - vin_learning_rate * vin_db1
vin_w2 = vin_w2 - vin_learning_rate * vin_dw2
vin_b2 = vin_b2 - vin_learning_rate * vin_db2
vin_w3 = vin_w3 - vin_learning_rate * vin_dw3
vin_b3 = vin_b3 - vin_learning_rate * vin_db3
print("All weights and biases updated with SGD.")
