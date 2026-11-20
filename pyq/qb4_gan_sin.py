# B4: GAN using sinusoidal data

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Generate real data (x, sin(x))
# --------------------------------------------------

vin_x = np.linspace(0, 2*np.pi, 1024)

vin_y = np.sin(vin_x)

vin_data = np.column_stack((vin_x, vin_y))

vin_data = torch.tensor(vin_data, dtype=torch.float32)

# Plot real data
plt.scatter(vin_data[:, 0], vin_data[:, 1])
plt.title("Real Sinusoidal Data")
plt.show()

# --------------------------------------------------
# 2. Generator
# --------------------------------------------------

class VinGenerator(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)

# --------------------------------------------------
# 3. Discriminator
# --------------------------------------------------

class VinDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

vin_gen = VinGenerator()
vin_disc = VinDiscriminator()

# Loss and optimizers
vin_loss = nn.BCELoss()

vin_g_optimizer = optim.Adam(vin_gen.parameters(), lr=0.001)
vin_d_optimizer = optim.Adam(vin_disc.parameters(), lr=0.001)

# --------------------------------------------------
# 4. Train GAN for 300 epochs
# --------------------------------------------------

for epoch in range(300):

    # Real data
    vin_real = vin_data

    # Generate fake data
    vin_noise = torch.randn(1024, 2)
    vin_fake = vin_gen(vin_noise)

    # Labels
    vin_real_labels = torch.ones(1024, 1)
    vin_fake_labels = torch.zeros(1024, 1)

    # Train discriminator
    vin_real_output = vin_disc(vin_real)
    vin_fake_output = vin_disc(vin_fake.detach())

    vin_d_loss_real = vin_loss(vin_real_output, vin_real_labels)
    vin_d_loss_fake = vin_loss(vin_fake_output, vin_fake_labels)

    vin_d_loss = vin_d_loss_real + vin_d_loss_fake

    vin_d_optimizer.zero_grad()
    vin_d_loss.backward()
    vin_d_optimizer.step()

    # Train generator
    vin_noise = torch.randn(1024, 2)
    vin_fake = vin_gen(vin_noise)

    vin_output = vin_disc(vin_fake)

    vin_g_loss = vin_loss(vin_output, vin_real_labels)

    vin_g_optimizer.zero_grad()
    vin_g_loss.backward()
    vin_g_optimizer.step()

    # Print loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print("Epoch:", epoch+1,
              "D Loss:", vin_d_loss.item(),
              "G Loss:", vin_g_loss.item())

# --------------------------------------------------
# 5. Generate 100 new samples
# --------------------------------------------------

vin_noise = torch.randn(100, 2)

with torch.no_grad():
    vin_generated = vin_gen(vin_noise)

# Plot generated data
plt.scatter(vin_generated[:, 0], vin_generated[:, 1])
plt.title("Generated Samples")
plt.show()

print("Generated samples:", vin_generated.shape)
