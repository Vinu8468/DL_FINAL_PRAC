# B1: CNN on CIFAR-10 + comparison with ResNet18

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Use GPU if available, otherwise CPU
vin_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", vin_device)

# --------------------------------------------------
# 1. Load CIFAR-10 dataset
# --------------------------------------------------

vin_transform = transforms.ToTensor()

vin_train_data = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=vin_transform
)

vin_test_data = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=vin_transform
)

vin_train_loader = DataLoader(vin_train_data, batch_size=64, shuffle=True)
vin_test_loader = DataLoader(vin_test_data, batch_size=64, shuffle=False)

# --------------------------------------------------
# 2. Define CNN with 3 convolutional layers
# --------------------------------------------------

class VinCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.pool(torch.relu(self.conv1(x)))  # 32 -> 16
        x = self.pool(torch.relu(self.conv2(x)))  # 16 -> 8
        x = torch.relu(self.conv3(x))             # Remains 8 x 8

        x = x.view(-1, 64 * 8 * 8)               # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Create model
vin_model = VinCNN().to(vin_device)

# Loss and optimizer
vin_loss = nn.CrossEntropyLoss()
vin_optimizer = optim.Adam(vin_model.parameters(), lr=0.001)

# --------------------------------------------------
# 3. Train CNN for 5 epochs
# --------------------------------------------------

for epoch in range(5):

    vin_total_loss = 0
    vin_correct = 0
    vin_total = 0

    vin_model.train()

    for vin_images, vin_labels in vin_train_loader:

        vin_images = vin_images.to(vin_device)
        vin_labels = vin_labels.to(vin_device)

        # Forward pass
        vin_outputs = vin_model(vin_images)
        vin_l = vin_loss(vin_outputs, vin_labels)

        # Backward pass and update weights
        vin_optimizer.zero_grad()
        vin_l.backward()
        vin_optimizer.step()

        vin_total_loss += vin_l.item()

        # Calculate training accuracy
        vin_pred = torch.argmax(vin_outputs, dim=1)
        vin_correct += (vin_pred == vin_labels).sum().item()
        vin_total += len(vin_labels)

    vin_acc = 100 * vin_correct / vin_total

    print("CNN Epoch:", epoch+1,
          "Loss:", vin_total_loss,
          "Accuracy:", vin_acc)

# --------------------------------------------------
# 4. Test CNN
# --------------------------------------------------

vin_correct = 0
vin_total = 0

vin_model.eval()

with torch.no_grad():

    for vin_images, vin_labels in vin_test_loader:

        vin_images = vin_images.to(vin_device)
        vin_labels = vin_labels.to(vin_device)

        vin_outputs = vin_model(vin_images)
        vin_pred = torch.argmax(vin_outputs, dim=1)

        vin_correct += (vin_pred == vin_labels).sum().item()
        vin_total += len(vin_labels)

vin_cnn_accuracy = 100 * vin_correct / vin_total

print("CNN Test Accuracy:", vin_cnn_accuracy)

# --------------------------------------------------
# 5. Load pretrained ResNet18
# --------------------------------------------------

# Resize images and normalize for pretrained ResNet
vin_transform_resnet = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

vin_train_resnet = datasets.CIFAR10(
    root='./data', train=True, download=True,
    transform=vin_transform_resnet
)

vin_test_resnet = datasets.CIFAR10(
    root='./data', train=False, download=True,
    transform=vin_transform_resnet
)

vin_train_loader_resnet = DataLoader(
    vin_train_resnet, batch_size=32, shuffle=True
)

vin_test_loader_resnet = DataLoader(
    vin_test_resnet, batch_size=32, shuffle=False
)

# Load pretrained ResNet18
vin_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Replace final layer for 10 CIFAR-10 classes
vin_resnet.fc = nn.Linear(vin_resnet.fc.in_features, 10)

vin_resnet = vin_resnet.to(vin_device)

vin_loss2 = nn.CrossEntropyLoss()
vin_optimizer2 = optim.Adam(vin_resnet.parameters(), lr=0.001)

# --------------------------------------------------
# 6. Train ResNet18 for 5 epochs
# --------------------------------------------------

for epoch in range(5):

    vin_total_loss = 0

    vin_resnet.train()

    for vin_images, vin_labels in vin_train_loader_resnet:

        vin_images = vin_images.to(vin_device)
        vin_labels = vin_labels.to(vin_device)

        vin_outputs = vin_resnet(vin_images)
        vin_l = vin_loss2(vin_outputs, vin_labels)

        vin_optimizer2.zero_grad()
        vin_l.backward()
        vin_optimizer2.step()

        vin_total_loss += vin_l.item()

    print("ResNet Epoch:", epoch+1,
          "Loss:", vin_total_loss)

# --------------------------------------------------
# 7. Test ResNet18
# --------------------------------------------------

vin_correct = 0
vin_total = 0

vin_resnet.eval()

with torch.no_grad():

    for vin_images, vin_labels in vin_test_loader_resnet:

        vin_images = vin_images.to(vin_device)
        vin_labels = vin_labels.to(vin_device)

        vin_outputs = vin_resnet(vin_images)
        vin_pred = torch.argmax(vin_outputs, dim=1)

        vin_correct += (vin_pred == vin_labels).sum().item()
        vin_total += len(vin_labels)

vin_resnet_accuracy = 100 * vin_correct / vin_total

print("ResNet18 Test Accuracy:", vin_resnet_accuracy)

# --------------------------------------------------
# 8. Compare both models
# --------------------------------------------------

print("\n--- Model Comparison ---")
print("CNN Accuracy:", vin_cnn_accuracy)
print("ResNet18 Accuracy:", vin_resnet_accuracy)
