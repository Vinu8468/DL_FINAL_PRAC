"""Lab 9: Simulate vanishing and exploding gradients across many layers."""
import numpy as np
import matplotlib.pyplot as plt

vin_num_layers = 30
vin_initial_gradient = 1.0
vin_vanishing = [vin_initial_gradient]
vin_exploding = [vin_initial_gradient]

for vin_layer in range(vin_num_layers):
    vin_vanishing.append(vin_vanishing[-1] * 0.5)
    vin_exploding.append(vin_exploding[-1] * 1.5)

plt.figure(figsize=(8, 4))
plt.plot(vin_vanishing, label="Vanishing gradient (multiply by 0.5)")
plt.plot(vin_exploding, label="Exploding gradient (multiply by 1.5)")
plt.yscale("log")
plt.xlabel("Number of layers")
plt.ylabel("Gradient magnitude (log scale)")
plt.title("Vanishing and exploding gradients")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---------- A second simulation: product of random derivatives
np.random.seed(4)
vin_small_derivatives = np.random.uniform(0.1, 0.9, 30)
vin_large_derivatives = np.random.uniform(1.1, 1.9, 30)
vin_small_product = np.cumprod(vin_small_derivatives)
vin_large_product = np.cumprod(vin_large_derivatives)
print("Final vanishing product:", vin_small_product[-1])
print("Final exploding product:", vin_large_product[-1])
