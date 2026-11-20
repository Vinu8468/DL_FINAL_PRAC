"""Lab 1: Activation functions and derivatives from scratch."""
import numpy as np
import matplotlib.pyplot as plt

# ---------- Input values
vin_z = np.linspace(-10, 10, 100)

# ---------- Sigmoid and derivative
def vin_sigmoid(vin_x):
    return 1.0 / (1.0 + np.exp(-vin_x))

def vin_sigmoid_derivative(vin_x):
    vin_a = vin_sigmoid(vin_x)
    return vin_a * (1.0 - vin_a)

# ---------- Tanh and derivative
def vin_tanh(vin_x):
    return (np.exp(vin_x) - np.exp(-vin_x)) / (np.exp(vin_x) + np.exp(-vin_x))

def vin_tanh_derivative(vin_x):
    vin_a = vin_tanh(vin_x)
    return 1.0 - vin_a ** 2

# ---------- ReLU and derivative
def vin_relu(vin_x):
    return np.maximum(0, vin_x)

def vin_relu_derivative(vin_x):
    return np.where(vin_x > 0, 1.0, 0.0)

# ---------- Leaky ReLU and derivative
def vin_leaky_relu(vin_x, vin_alpha=0.01):
    return np.where(vin_x > 0, vin_x, vin_alpha * vin_x)

def vin_leaky_relu_derivative(vin_x, vin_alpha=0.01):
    return np.where(vin_x > 0, 1.0, vin_alpha)

# ---------- Softmax (stable version)
def vin_softmax(vin_x):
    vin_shifted = vin_x - np.max(vin_x)
    vin_exp = np.exp(vin_shifted)
    return vin_exp / np.sum(vin_exp)

vin_functions = [
    ("Sigmoid", vin_sigmoid, vin_sigmoid_derivative),
    ("Tanh", vin_tanh, vin_tanh_derivative),
    ("ReLU", vin_relu, vin_relu_derivative),
    ("Leaky ReLU", vin_leaky_relu, vin_leaky_relu_derivative)
]

# ---------- Plot functions and derivatives
for vin_name, vin_function, vin_derivative in vin_functions:
    vin_output = vin_function(vin_z)
    vin_gradient = vin_derivative(vin_z)
    print(vin_name, "minimum:", np.min(vin_output), "maximum:", np.max(vin_output))
    print(vin_name, "zero-centred:", bool(np.min(vin_output) < 0 and np.max(vin_output) > 0))
    print(vin_name, "gradient at z=-10, z=0, z=10:", vin_gradient[0], vin_gradient[50], vin_gradient[-1])

    plt.figure(figsize=(7, 4))
    plt.plot(vin_z, vin_output, label=vin_name)
    plt.plot(vin_z, vin_gradient, label="Derivative")
    plt.title(vin_name + " and its derivative")
    plt.xlabel("z")
    plt.ylabel("Output")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

print("Softmax example:", vin_softmax(np.array([2.0, 1.0, 0.1])))
print("Observation: sigmoid output is 0 to 1; tanh is -1 to 1 and zero-centred.")
print("For large positive or negative inputs, sigmoid and tanh gradients approach zero.")
print("Tanh(z) = 2 x sigmoid(2z) - 1.")
