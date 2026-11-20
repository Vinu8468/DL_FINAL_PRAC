import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

vin_transform = transforms.ToTensor()

vin_train_data = datasets.MNIST('./data', train=True, download=True, transform=vin_transform)
vin_test_data = datasets.MNIST('./data', train=False, download=True, transform=vin_transform)

vin_train_loader = DataLoader(vin_train_data, batch_size=64, shuffle=True)
vin_test_loader = DataLoader(vin_test_data, batch_size=64, shuffle=False)

class VinNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

vin_model = VinNet()
vin_loss = nn.CrossEntropyLoss()
vin_optimizer = optim.Adam(vin_model.parameters(), lr=0.001)

for epoch in range(5):
    vin_total_loss = 0

    for vin_images, vin_labels in vin_train_loader:
        vin_outputs = vin_model(vin_images)
        vin_l = vin_loss(vin_outputs, vin_labels)

        vin_optimizer.zero_grad()
        vin_l.backward()
        vin_optimizer.step()

        vin_total_loss += vin_l.item()

    print("Epoch", epoch+1, "Loss:", vin_total_loss)

vin_correct = 0
vin_total = 0

with torch.no_grad():
    for vin_images, vin_labels in vin_test_loader:
        vin_outputs = vin_model(vin_images)
        vin_predicted = torch.argmax(vin_outputs, dim=1)

        vin_correct += (vin_predicted == vin_labels).sum().item()
        vin_total += len(vin_labels)

print("Test accuracy:", 100 * vin_correct / vin_total, "%")
