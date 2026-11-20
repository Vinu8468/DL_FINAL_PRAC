"""Lab 12: CNN image classification using MNIST or CIFAR10.

Default is MNIST. Set vin_dataset_name = "CIFAR10" to use CIFAR10.
The first run downloads the selected dataset using torchvision.
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

vin_dataset_name = "MNIST"   # change to "CIFAR10" when needed
vin_batch_size = 64
vin_epochs = 2
vin_train_limit = 5000       # set None to use the complete training set
vin_test_limit = 1000
vin_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Dataset and transforms
if vin_dataset_name == "MNIST":
    vin_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    vin_train_dataset = datasets.MNIST("data", train=True, download=True, transform=vin_transform)
    vin_test_dataset = datasets.MNIST("data", train=False, download=True, transform=vin_transform)
    vin_input_channels = 1
else:
    vin_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    vin_train_dataset = datasets.CIFAR10("data", train=True, download=True, transform=vin_transform)
    vin_test_dataset = datasets.CIFAR10("data", train=False, download=True, transform=vin_transform)
    vin_input_channels = 3

# ---------- Small subsets make the practical faster on CPU
if vin_train_limit is not None:
    vin_train_dataset = Subset(vin_train_dataset, range(min(vin_train_limit, len(vin_train_dataset))))
if vin_test_limit is not None:
    vin_test_dataset = Subset(vin_test_dataset, range(min(vin_test_limit, len(vin_test_dataset))))
vin_train_loader = DataLoader(vin_train_dataset, batch_size=vin_batch_size, shuffle=True)
vin_test_loader = DataLoader(vin_test_dataset, batch_size=vin_batch_size, shuffle=False)

class VinCNN(nn.Module):
    def __init__(self, vin_channels, vin_depth):
        super().__init__()
        vin_layers = []
        vin_current_channels = vin_channels
        for vin_layer in range(vin_depth):
            vin_next_channels = 16 * (vin_layer + 1)
            vin_layers.append(nn.Conv2d(vin_current_channels, vin_next_channels, 3, padding=1))
            vin_layers.append(nn.ReLU())
            if vin_layer == 0:
                vin_layers.append(nn.MaxPool2d(2))
            vin_current_channels = vin_next_channels
        self.features = nn.ModuleList(vin_layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(vin_current_channels, 10)

    def forward(self, vin_input):
        vin_features = vin_input
        for vin_layer in self.features:
            vin_features = vin_layer(vin_features)
        vin_features = self.pool(vin_features)
        vin_features = torch.flatten(vin_features, 1)
        return self.classifier(vin_features)

# ---------- Train networks with increasing convolution depth and compare train errors
vin_depth_values = []
vin_train_errors = []
for vin_depth in [1, 2, 3]:
    vin_model = VinCNN(vin_input_channels, vin_depth).to(vin_device)
    vin_optimizer = torch.optim.Adam(vin_model.parameters(), lr=0.001)
    vin_loss_function = nn.CrossEntropyLoss()

    for vin_epoch in range(vin_epochs):
        vin_model.train()
        vin_total_loss = 0.0
        vin_total_correct = 0
        vin_total = 0
        for vin_images, vin_labels in vin_train_loader:
            vin_images = vin_images.to(vin_device)
            vin_labels = vin_labels.to(vin_device)
            vin_output = vin_model(vin_images)
            vin_loss = vin_loss_function(vin_output, vin_labels)
            vin_optimizer.zero_grad()
            vin_loss.backward()
            vin_optimizer.step()
            vin_total_loss = vin_total_loss + vin_loss.item()
            vin_total_correct = vin_total_correct + (torch.argmax(vin_output, 1) == vin_labels).sum().item()
            vin_total = vin_total + len(vin_labels)
        print(vin_dataset_name, "depth", vin_depth, "epoch", vin_epoch + 1,
              "loss", round(vin_total_loss / len(vin_train_loader), 4),
              "training accuracy", round(vin_total_correct / vin_total, 4))

    vin_model.eval()
    vin_train_correct = 0
    vin_train_total = 0
    with torch.no_grad():
        for vin_images, vin_labels in vin_train_loader:
            vin_images = vin_images.to(vin_device)
            vin_labels = vin_labels.to(vin_device)
            vin_train_output = vin_model(vin_images)
            vin_train_correct = vin_train_correct + (torch.argmax(vin_train_output, 1) == vin_labels).sum().item()
            vin_train_total = vin_train_total + len(vin_labels)
    vin_train_error = 1.0 - vin_train_correct / vin_train_total
    vin_depth_values.append(vin_depth)
    vin_train_errors.append(vin_train_error)
    print("Depth", vin_depth, "final training error:", round(vin_train_error, 4))

    vin_correct = 0
    vin_total = 0
    with torch.no_grad():
        for vin_images, vin_labels in vin_test_loader:
            vin_images = vin_images.to(vin_device)
            vin_labels = vin_labels.to(vin_device)
            vin_output = vin_model(vin_images)
            vin_correct = vin_correct + (torch.argmax(vin_output, 1) == vin_labels).sum().item()
            vin_total = vin_total + len(vin_labels)
    print("Depth", vin_depth, "test accuracy:", round(vin_correct / vin_total, 4))
    torch.save(vin_model.state_dict(), "12_" + vin_dataset_name.lower() + "_cnn_depth" + str(vin_depth) + ".pth")

# ---------- Plot training error against network depth
plt.plot(vin_depth_values, vin_train_errors, marker="o")
plt.xlabel("Number of convolution layers")
plt.ylabel("Training error (1 - accuracy)")
plt.title("Training error vs CNN depth")
plt.grid(True)
plt.show()
