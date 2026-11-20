"""Lab 20 (additional): Self-supervised image representation learning with SimCLR.

Two augmented views of every image are made. The CNN learns to bring the two views
of the same image closer in embedding space and separate other images in the batch.
This uses CIFAR10 from torchvision; set vin_train_limit to a small number for CPU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

vin_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vin_train_limit = 3000
vin_epochs = 3

# ---------- Two stochastic transformations create positive pairs
vin_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.7),
    transforms.ToTensor()
])

class VinTwoViewDataset(torch.utils.data.Dataset):
    def __init__(self, vin_dataset):
        self.dataset = vin_dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, vin_index):
        vin_image, vin_label = self.dataset[vin_index]
        vin_view1 = vin_transform(vin_image)
        vin_view2 = vin_transform(vin_image)
        return vin_view1, vin_view2

vin_base_dataset = datasets.CIFAR10("data", train=True, download=True, transform=None)
if vin_train_limit is not None:
    vin_base_dataset = Subset(vin_base_dataset, range(min(vin_train_limit, len(vin_base_dataset))))
vin_dataset = VinTwoViewDataset(vin_base_dataset)
vin_loader = DataLoader(vin_dataset, batch_size=64, shuffle=True, drop_last=True)

class VinSimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(64, 128)
        )
        self.projector = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64))
    def forward(self, vin_input):
        vin_features = self.encoder(vin_input)
        vin_projection = self.projector(vin_features)
        return F.normalize(vin_projection, dim=1)

def vin_nt_xent(vin_z1, vin_z2, vin_temperature=0.5):
    vin_batch_size = vin_z1.shape[0]
    vin_all = torch.cat((vin_z1, vin_z2), dim=0)
    vin_similarity = torch.matmul(vin_all, vin_all.T) / vin_temperature
    vin_mask = torch.eye(vin_batch_size * 2, device=vin_all.device).bool()
    vin_similarity = vin_similarity.masked_fill(vin_mask, -1e9)
    vin_positive = torch.cat((
        torch.diag(vin_similarity, vin_batch_size),
        torch.diag(vin_similarity, -vin_batch_size)
    ), dim=0)
    vin_denominator = torch.logsumexp(vin_similarity, dim=1)
    return torch.mean(-vin_positive + vin_denominator)

vin_model = VinSimCLR().to(vin_device)
vin_optimizer = torch.optim.Adam(vin_model.parameters(), lr=0.001)
for vin_epoch in range(vin_epochs):
    vin_model.train()
    vin_total_loss = 0.0
    for vin_view1, vin_view2 in vin_loader:
        vin_view1 = vin_view1.to(vin_device)
        vin_view2 = vin_view2.to(vin_device)
        vin_z1 = vin_model(vin_view1)
        vin_z2 = vin_model(vin_view2)
        vin_loss = vin_nt_xent(vin_z1, vin_z2)
        vin_optimizer.zero_grad()
        vin_loss.backward()
        vin_optimizer.step()
        vin_total_loss = vin_total_loss + vin_loss.item()
    print("Epoch", vin_epoch + 1, "contrastive loss", vin_total_loss / len(vin_loader))

torch.save(vin_model.encoder.state_dict(), "20_simclr_encoder.pth")
print("Saved the learned image encoder.")

# ---------- Linear probe: train a small classifier on frozen self-supervised features
# This demonstrates downstream image classification using the learned representation.
vin_eval_transform = transforms.ToTensor()
vin_linear_train_base = datasets.CIFAR10("data", train=True, download=True, transform=vin_eval_transform)
vin_linear_test_base = datasets.CIFAR10("data", train=False, download=True, transform=vin_eval_transform)
vin_linear_train_subset = Subset(vin_linear_train_base, range(min(vin_train_limit, len(vin_linear_train_base))))
vin_linear_test_subset = Subset(vin_linear_test_base, range(min(1000, len(vin_linear_test_base))))
vin_linear_train_loader = DataLoader(vin_linear_train_subset, batch_size=64, shuffle=False)
vin_linear_test_loader = DataLoader(vin_linear_test_subset, batch_size=64, shuffle=False)

vin_model.eval()
vin_train_features = []
vin_train_labels = []
vin_test_features = []
vin_test_labels = []
with torch.no_grad():
    for vin_images, vin_labels in vin_linear_train_loader:
        vin_features = vin_model.encoder(vin_images.to(vin_device)).cpu()
        vin_train_features.append(vin_features)
        vin_train_labels.append(vin_labels)
    for vin_images, vin_labels in vin_linear_test_loader:
        vin_features = vin_model.encoder(vin_images.to(vin_device)).cpu()
        vin_test_features.append(vin_features)
        vin_test_labels.append(vin_labels)

vin_train_feature_tensor = torch.cat(vin_train_features, dim=0)
vin_train_label_tensor = torch.cat(vin_train_labels, dim=0)
vin_test_feature_tensor = torch.cat(vin_test_features, dim=0)
vin_test_label_tensor = torch.cat(vin_test_labels, dim=0)
vin_probe_loader = DataLoader(
    TensorDataset(vin_train_feature_tensor, vin_train_label_tensor),
    batch_size=64, shuffle=True
)
vin_linear_probe = nn.Linear(128, 10)
vin_probe_optimizer = torch.optim.Adam(vin_linear_probe.parameters(), lr=0.01)
vin_probe_loss = nn.CrossEntropyLoss()
for vin_epoch in range(3):
    for vin_features, vin_labels in vin_probe_loader:
        vin_logits = vin_linear_probe(vin_features)
        vin_loss = vin_probe_loss(vin_logits, vin_labels)
        vin_probe_optimizer.zero_grad()
        vin_loss.backward()
        vin_probe_optimizer.step()
    print("Linear probe epoch", vin_epoch + 1, "loss", vin_loss.item())
with torch.no_grad():
    vin_test_logits = vin_linear_probe(vin_test_feature_tensor)
    vin_test_accuracy = (torch.argmax(vin_test_logits, dim=1) == vin_test_label_tensor).float().mean().item()
print("Linear-probe test accuracy:", round(vin_test_accuracy, 4))
torch.save(vin_linear_probe.state_dict(), "20_simclr_linear_probe.pth")
