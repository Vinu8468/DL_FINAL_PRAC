# pip install numpy pandas matplotlib scikit-learn torch torchvision


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn

# --------------------------------------------------
# 1. TEXT REPRESENTATION LEARNING
# --------------------------------------------------

vin_text = [
    "gene expression analysis identifies disease genes",
    "genomics studies DNA and gene expression",
    "deep learning models learn data representations",
    "machine learning predicts biological outcomes",
    "protein sequences contain biological information",
    "DNA sequences contain genetic information"
]

# TF-IDF representation
vin_tfidf = TfidfVectorizer()

vin_text_features = vin_tfidf.fit_transform(vin_text)

print("TF-IDF Shape:", vin_text_features.shape)
print(vin_text_features.toarray())

# Reduce dimensions using Truncated SVD
vin_svd = TruncatedSVD(n_components=2)

vin_text_reduced = vin_svd.fit_transform(vin_text_features)

print("Reduced Text Shape:", vin_text_reduced.shape)

# Visualize
plt.scatter(vin_text_reduced[:, 0], vin_text_reduced[:, 1])

for vin_i in range(len(vin_text)):
    plt.text(
        vin_text_reduced[vin_i, 0],
        vin_text_reduced[vin_i, 1],
        str(vin_i)
    )

plt.title("Text Representation using TF-IDF + SVD")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()


# --------------------------------------------------
# 2. IMAGE REPRESENTATION LEARNING
# --------------------------------------------------

from sklearn.datasets import load_digits

vin_digits = load_digits()

vin_images = vin_digits.images
vin_labels = vin_digits.target

# Convert images into flattened vectors
vin_image_features = vin_images.reshape(
    vin_images.shape[0],
    -1
)

print("Image Feature Shape:", vin_image_features.shape)

# PCA representation
vin_pca = PCA(n_components=2)

vin_image_reduced = vin_pca.fit_transform(vin_image_features)

print("PCA Shape:", vin_image_reduced.shape)

# Visualize
plt.scatter(
    vin_image_reduced[:, 0],
    vin_image_reduced[:, 1],
    c=vin_labels
)

plt.title("Image Representation using PCA")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar()
plt.show()


# --------------------------------------------------
# 3. BIOLOGICAL DATA REPRESENTATION LEARNING
# --------------------------------------------------

# Simulate gene expression data
# Rows = samples
# Columns = genes

vin_bio_data = np.random.rand(100, 500)

vin_bio_labels = np.random.randint(0, 2, 100)

print("Biological Data Shape:", vin_bio_data.shape)

# Standardize the data
vin_scaler = StandardScaler()

vin_bio_scaled = vin_scaler.fit_transform(vin_bio_data)

# PCA
vin_bio_pca = PCA(n_components=2)

vin_bio_reduced = vin_bio_pca.fit_transform(vin_bio_scaled)

print("Biological PCA Shape:", vin_bio_reduced.shape)

# Visualize
plt.scatter(
    vin_bio_reduced[:, 0],
    vin_bio_reduced[:, 1],
    c=vin_bio_labels
)

plt.title("Biological Data Representation using PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar()
plt.show()


# --------------------------------------------------
# 4. AUTOENCODER REPRESENTATION LEARNING
# --------------------------------------------------

# Using biological dataset

vin_input_size = vin_bio_scaled.shape[1]
vin_latent_size = 2

# Convert to tensor
vin_x = torch.tensor(
    vin_bio_scaled,
    dtype=torch.float32
)

# Autoencoder
class VinAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(vin_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, vin_latent_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(vin_latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, vin_input_size)
        )

    def forward(self, vin_input):

        vin_encoded = self.encoder(vin_input)

        vin_decoded = self.decoder(vin_encoded)

        return vin_decoded


vin_autoencoder = VinAutoencoder()

vin_optimizer = torch.optim.Adam(
    vin_autoencoder.parameters(),
    lr=0.001
)

vin_loss_function = nn.MSELoss()

# Training
for vin_epoch in range(100):

    vin_output = vin_autoencoder(vin_x)

    vin_loss = vin_loss_function(
        vin_output,
        vin_x
    )

    vin_optimizer.zero_grad()

    vin_loss.backward()

    vin_optimizer.step()

    if vin_epoch % 10 == 0:
        print(
            "Epoch:",
            vin_epoch,
            "Loss:",
            vin_loss.item()
        )


# Extract learned representation
with torch.no_grad():

    vin_encoded = vin_autoencoder.encoder(vin_x)

    vin_encoded = vin_encoded.numpy()

print("Autoencoder Representation:", vin_encoded.shape)

# Visualize latent representation
plt.scatter(
    vin_encoded[:, 0],
    vin_encoded[:, 1],
    c=vin_bio_labels
)

plt.title("Biological Representation using Autoencoder")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.colorbar()
plt.show()

