# A4: Autoencoder forward pass

import torch
import torch.nn as nn

# Create Autoencoder
class VinAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder: 784 -> 128 -> 32
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )

        # Decoder: 32 -> 128 -> 784
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.ReLU()
        )

    def forward(self, x):
        vin_latent = self.encoder(x)
        vin_output = self.decoder(vin_latent)
        return vin_output

# Create model
vin_model = VinAutoencoder()

# Generate random input of size 784
vin_x = torch.randn(1, 784)

# Forward pass
vin_output = vin_model(vin_x)

print("Input shape:", vin_x.shape)
print("Output shape:", vin_output.shape)
