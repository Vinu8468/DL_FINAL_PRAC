# B3: Denoising Autoencoder using MNIST

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load MNIST
vin_transform = transforms.ToTensor()

vin_train_data = datasets.MNIST(
    './data', train=True, download=True, transform=vin_transform
)

vin_test_data = datasets.MNIST(
    './data', train=False, download=True, transform=vin_transform
)

vin_train_loader = DataLoader(vin_train_data, batch_size=64, shuffle=True)
vin_test_loader = DataLoader(vin_test_data, batch_size=64, shuffle=False)

# Autoencoder model
class VinAutoencoder(nn.Module):

    def __init__(self, bottleneck):
        super().__init__()

        # Encoder compresses image
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, bottleneck)
        )

        # Decoder reconstructs image
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Try different bottleneck sizes
for vin_size in [32, 64, 128]:

    vin_model = VinAutoencoder(vin_size)
    vin_optimizer = optim.Adam(vin_model.parameters(), lr=0.001)
    vin_loss = nn.MSELoss()

    # Train
    for epoch in range(5):

        vin_total_loss = 0

        for vin_images, vin_labels in vin_train_loader:

            # Add noise to input
            vin_noisy = vin_images + 0.3 * torch.randn_like(vin_images)
            vin_noisy = torch.clamp(vin_noisy, 0, 1)

            vin_outputs = vin_model(vin_noisy)
            vin_l = vin_loss(vin_outputs, vin_images.view(-1, 784))

            vin_optimizer.zero_grad()
            vin_l.backward()
            vin_optimizer.step()

            vin_total_loss += vin_l.item()

        print("Bottleneck:", vin_size,
              "Epoch:", epoch+1,
              "Loss:", vin_total_loss)

    # Test reconstruction
    vin_images, vin_labels = next(iter(vin_test_loader))

    vin_noisy = vin_images + 0.3 * torch.randn_like(vin_images)
    vin_noisy = torch.clamp(vin_noisy, 0, 1)

    with torch.no_grad():
        vin_reconstructed = vin_model(vin_noisy)

    # Display original, noisy and reconstructed images
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(vin_images[0].squeeze(), cmap='gray')
    plt.title("Original")

    plt.subplot(1, 3, 2)
    plt.imshow(vin_noisy[0].squeeze(), cmap='gray')
    plt.title("Noisy")

    plt.subplot(1, 3, 3)
    plt.imshow(vin_reconstructed[0].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")

    plt.show()
