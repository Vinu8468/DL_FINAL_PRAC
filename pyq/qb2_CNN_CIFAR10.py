# B2: CNN classification using CIFAR-10

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load CIFAR-10 dataset
vin_transform = transforms.ToTensor()

vin_train_data = datasets.CIFAR10(
    './data', train=True, download=True, transform=vin_transform
)

vin_test_data = datasets.CIFAR10(
    './data', train=False, download=True, transform=vin_transform
)

vin_train_loader = DataLoader(vin_train_data, batch_size=64, shuffle=True)
vin_test_loader = DataLoader(vin_test_data, batch_size=64, shuffle=False)

# Define CNN: CONV -> CONV -> MAXPOOL -> LINEAR -> LINEAR
class VinCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = torch.relu(self.conv1(x))  # First convolution
        x = torch.relu(self.conv2(x))  # Second convolution
        x = self.pool(x)              # Reduce image size

        x = x.view(-1, 32 * 16 * 16) # Flatten
        x = torch.relu(self.fc1(x))   # First linear layer
        x = self.fc2(x)               # Output layer

        return x

# Create model
vin_model = VinCNN()

# Loss and optimizer
vin_loss = nn.CrossEntropyLoss()
vin_optimizer = optim.Adam(vin_model.parameters(), lr=0.001)

# Train for 5 epochs
for epoch in range(5):

    vin_total_loss = 0

    for vin_images, vin_labels in vin_train_loader:

        vin_outputs = vin_model(vin_images)
        vin_l = vin_loss(vin_outputs, vin_labels)

        vin_optimizer.zero_grad()
        vin_l.backward()
        vin_optimizer.step()

        vin_total_loss += vin_l.item()

    print("Epoch:", epoch+1, "Loss:", vin_total_loss)

# Test model
vin_correct = 0
vin_total = 0

with torch.no_grad():

    for vin_images, vin_labels in vin_test_loader:

        vin_outputs = vin_model(vin_images)
        vin_pred = torch.argmax(vin_outputs, dim=1)

        vin_correct += (vin_pred == vin_labels).sum().item()
        vin_total += len(vin_labels)

print("Test accuracy:", 100 * vin_correct / vin_total, "%")
