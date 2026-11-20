# A1: Activation functions and their derivatives from scratch

import numpy as np
import matplotlib.pyplot as plt
import math

# Generate 100 values between -10 and 10
vin_z = np.linspace(-10, 10, 100)

# Activation functions
def vin_sigmoid(z):
    return 1 / (1 + math.exp(-z))

def vin_tanh(z):
    return (math.exp(z) - math.exp(-z)) / (math.exp(z) + math.exp(-z))

def vin_relu(z):
    if z > 0:
        return z
    else:
        return 0

# Derivatives
def vin_sigmoid_derivative(z):
    vin_s = vin_sigmoid(z)
    return vin_s * (1 - vin_s)

def vin_tanh_derivative(z):
    vin_t = vin_tanh(z)
    return 1 - vin_t * vin_t

def vin_relu_derivative(z):
    if z > 0:
        return 1
    else:
        return 0

# Calculate outputs using loops
vin_sig = []
vin_sig_d = []
vin_t = []
vin_t_d = []
vin_r = []
vin_r_d = []

for vin_i in vin_z:

    vin_sig.append(vin_sigmoid(vin_i))
    vin_sig_d.append(vin_sigmoid_derivative(vin_i))

    vin_t.append(vin_tanh(vin_i))
    vin_t_d.append(vin_tanh_derivative(vin_i))

    vin_r.append(vin_relu(vin_i))
    vin_r_d.append(vin_relu_derivative(vin_i))

# Plot activation functions and derivatives
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(vin_z, vin_sig)
plt.title("Sigmoid")

plt.subplot(2, 3, 2)
plt.plot(vin_z, vin_t)
plt.title("Tanh")

plt.subplot(2, 3, 3)
plt.plot(vin_z, vin_r)
plt.title("ReLU")

plt.subplot(2, 3, 4)
plt.plot(vin_z, vin_sig_d)
plt.title("Sigmoid Derivative")

plt.subplot(2, 3, 5)
plt.plot(vin_z, vin_t_d)
plt.title("Tanh Derivative")

plt.subplot(2, 3, 6)
plt.plot(vin_z, vin_r_d)
plt.title("ReLU Derivative")

plt.tight_layout()
plt.show()
