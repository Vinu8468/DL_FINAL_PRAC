import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Parameters
vin_latent_dim = 10
vin_hidden_dim = 128
vin_batch_size = 64
vin_epochs = 5000
vin_lr = 0.0002

# Generator
class vin_Generator(nn.Module):
    def __init__(self):
        super(vin_Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(vin_latent_dim, vin_hidden_dim),
            nn.ReLU(),
            nn.Linear(vin_hidden_dim, vin_hidden_dim),
            nn.ReLU(),
            nn.Linear(vin_hidden_dim, 1)
        )

    def forward(self, vin_x):
        return self.model(vin_x)

# Discriminator
class vin_Discriminator(nn.Module):
    def __init__(self):
        super(vin_Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1, vin_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(vin_hidden_dim, vin_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(vin_hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, vin_x):
        return self.model(vin_x)

# Initialize models
vin_generator = vin_Generator()
vin_discriminator = vin_Discriminator()

vin_criterion = nn.BCELoss()

vin_g_optimizer = torch.optim.Adam(
    vin_generator.parameters(), lr=vin_lr
)

vin_d_optimizer = torch.optim.Adam(
    vin_discriminator.parameters(), lr=vin_lr
)

# Training GAN
for vin_epoch in range(vin_epochs):

    # Generate real Gaussian samples
    vin_real = torch.randn(vin_batch_size, 1)

    # Generate fake samples
    vin_noise = torch.randn(vin_batch_size, vin_latent_dim)
    vin_fake = vin_generator(vin_noise)

    # Train discriminator
    vin_real_labels = torch.ones(vin_batch_size, 1)
    vin_fake_labels = torch.zeros(vin_batch_size, 1)

    vin_d_real = vin_discriminator(vin_real)
    vin_d_fake = vin_discriminator(vin_fake.detach())

    vin_loss_real = vin_criterion(vin_d_real, vin_real_labels)
    vin_loss_fake = vin_criterion(vin_d_fake, vin_fake_labels)

    vin_d_loss = vin_loss_real + vin_loss_fake

    vin_d_optimizer.zero_grad()
    vin_d_loss.backward()
    vin_d_optimizer.step()

    # Train generator
    vin_noise = torch.randn(vin_batch_size, vin_latent_dim)
    vin_fake = vin_generator(vin_noise)

    vin_output = vin_discriminator(vin_fake)

    vin_g_loss = vin_criterion(vin_output, vin_real_labels)

    vin_g_optimizer.zero_grad()
    vin_g_loss.backward()
    vin_g_optimizer.step()

    if (vin_epoch + 1) % 500 == 0:
        print("Epoch:", vin_epoch + 1,
              "D Loss:", vin_d_loss.item(),
              "G Loss:", vin_g_loss.item())

# Generate samples
vin_noise = torch.randn(1000, vin_latent_dim)

with torch.no_grad():
    vin_generated = vin_generator(vin_noise)

# Plot real and generated data
vin_real_data = torch.randn(1000, 1)

plt.hist(vin_real_data.numpy(), bins=30, alpha=0.5,
         label="Real Data", density=True)

plt.hist(vin_generated.numpy(), bins=30, alpha=0.5,
         label="Generated Data", density=True)

plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Real vs Generated Gaussian Data")
plt.legend()
plt.show()
