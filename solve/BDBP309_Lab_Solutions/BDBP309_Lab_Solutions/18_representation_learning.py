"""Lab 18: Representation learning for text, image and biological-style data."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import torch
import torch.nn as nn

# --------------------------------------------------
# 1. TEXT: TF-IDF followed by SVD (latent semantic representation)
# --------------------------------------------------
vin_text = [
    "gene expression analysis identifies disease genes",
    "genomics studies DNA and gene expression",
    "deep learning models learn data representations",
    "machine learning predicts biological outcomes",
    "protein sequences contain biological information",
    "DNA sequences contain genetic information"
]
vin_vectorizer = TfidfVectorizer()
vin_text_matrix = vin_vectorizer.fit_transform(vin_text)
print("Text TF-IDF matrix:", vin_text_matrix.shape)
vin_text_svd = TruncatedSVD(n_components=2, random_state=42)
vin_text_embedding = vin_text_svd.fit_transform(vin_text_matrix)
plt.figure()
plt.scatter(vin_text_embedding[:, 0], vin_text_embedding[:, 1])
for vin_i in range(len(vin_text)):
    plt.text(vin_text_embedding[vin_i, 0], vin_text_embedding[vin_i, 1], str(vin_i))
plt.title("Text representation: TF-IDF + SVD")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

# Other text representation options:
# from sklearn.feature_extraction.text import CountVectorizer
# vin_count_matrix = CountVectorizer().fit_transform(vin_text)
# Word2Vec / FastText: train dense word embeddings with gensim (optional package).
# Transformers: use AutoTokenizer and AutoModel for contextual token embeddings.

# --------------------------------------------------
# 2. IMAGE: PCA on flattened handwritten digit images
# --------------------------------------------------
vin_digit_data = load_digits()
vin_images = vin_digit_data.images
vin_image_labels = vin_digit_data.target
vin_image_matrix = vin_images.reshape(vin_images.shape[0], -1)
vin_image_pca = PCA(n_components=2, random_state=42)
vin_image_embedding = vin_image_pca.fit_transform(vin_image_matrix)
print("Image PCA matrix:", vin_image_embedding.shape)
plt.figure()
plt.scatter(vin_image_embedding[:, 0], vin_image_embedding[:, 1], c=vin_image_labels, s=8)
plt.title("Image representation: PCA on digits")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar()
plt.show()

# Other image representation options:
# CNN: remove the final classification layer and use the hidden feature vector.
# Autoencoder: train encoder + decoder and use the encoder's latent vector.
# Self-supervised: learn image embeddings using SimCLR (see 20_self_supervised_learning.py).

# --------------------------------------------------
# 3. BIOLOGICAL: PCA and autoencoder on gene-expression-like data
# --------------------------------------------------
# Replace this generated matrix with a real samples x genes matrix from CSV/NPZ.
# Rows are samples; columns are genes.
if os.path.exists("gene_expression_matrix.csv"):
    # Supply one row per sample and one numeric column per gene.
    vin_bio_frame = pd.read_csv("gene_expression_matrix.csv")
    vin_bio_matrix = vin_bio_frame.select_dtypes(include=[np.number]).values
    print("Loaded gene_expression_matrix.csv")
else:
    np.random.seed(18)
    vin_bio_matrix = np.random.randn(120, 500)
    vin_bio_matrix[:, :10] = vin_bio_matrix[:, :10] + np.random.randn(120, 1) * 2
    print("Using synthetic gene-expression-like data; replace with gene_expression_matrix.csv for biological analysis.")
vin_bio_scaled = StandardScaler().fit_transform(vin_bio_matrix)
vin_bio_pca = PCA(n_components=2, random_state=42)
vin_bio_embedding = vin_bio_pca.fit_transform(vin_bio_scaled)
print("Biological PCA matrix:", vin_bio_embedding.shape)
plt.figure()
plt.scatter(vin_bio_embedding[:, 0], vin_bio_embedding[:, 1], s=12)
plt.title("Biological-style representation: PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# Autoencoder learns a nonlinear compressed latent representation.
vin_x = torch.tensor(vin_bio_scaled, dtype=torch.float32)
vin_input_size = vin_x.shape[1]

class VinAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(vin_input_size, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, vin_input_size)
        )
    def forward(self, vin_input):
        vin_latent = self.encoder(vin_input)
        vin_reconstructed = self.decoder(vin_latent)
        return vin_reconstructed

vin_autoencoder = VinAutoencoder()
vin_optimizer = torch.optim.Adam(vin_autoencoder.parameters(), lr=0.001)
vin_loss_function = nn.MSELoss()
for vin_epoch in range(100):
    vin_reconstruction = vin_autoencoder(vin_x)
    vin_loss = vin_loss_function(vin_reconstruction, vin_x)
    vin_optimizer.zero_grad()
    vin_loss.backward()
    vin_optimizer.step()
    if vin_epoch % 20 == 0:
        print("Autoencoder epoch", vin_epoch, "loss", vin_loss.item())
with torch.no_grad():
    vin_latent = vin_autoencoder.encoder(vin_x).numpy()
plt.figure()
plt.scatter(vin_latent[:, 0], vin_latent[:, 1], s=12)
plt.title("Biological-style representation: Autoencoder")
plt.xlabel("Latent dimension 1")
plt.ylabel("Latent dimension 2")
plt.show()
