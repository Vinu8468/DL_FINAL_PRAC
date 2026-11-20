# A2: Load CIFAR-10 and normalize images

import torch
from torchvision import datasets, transforms

# Convert images to tensors and normalize to [0,1]
vin_transform = transforms.ToTensor()

# Load training dataset
vin_train = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=vin_transform
)

# Load testing dataset
vin_test = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=vin_transform
)

# Print dataset sizes
print("Training samples:", len(vin_train))
print("Testing samples:", len(vin_test))

# Check one image
vin_image, vin_label = vin_train[0]

print("Image shape:", vin_image.shape)
print("Pixel minimum:", vin_image.min())
print("Pixel maximum:", vin_image.max())
print("Label:", vin_label)
