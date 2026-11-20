# A3: Single layer neural network forward pass

import torch
import torch.nn as nn

# Create a layer with 3 inputs and 1 output
vin_model = nn.Linear(3, 1)

# Input sample (3 features)
vin_x = torch.tensor([[2.0, 3.0, 4.0]])

# Forward pass
vin_output = vin_model(vin_x)

print("Input:", vin_x)
print("Output:", vin_output)
