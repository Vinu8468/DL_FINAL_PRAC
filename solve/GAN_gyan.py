
# ============================================================
# PRACTICAL: GENERATIVE ADVERSARIAL NETWORK (GAN)
# ============================================================
#
# AIM:
# Implement a GAN using PyTorch.
#
# A GAN contains two neural networks:
#
# 1. Generator:
#    Takes random noise as input and generates fake data.
#
# 2. Discriminator:
#    Takes real or fake data and predicts whether it is real.
#
# The Generator tries to fool the Discriminator.
# The Discriminator tries to identify fake samples.
#
# DATA:
# We will create a sinusoidal dataset:
# x = t / pi
# y = sin(t)
#
# The Generator will learn to produce points resembling
# the distribution of this real dataset.
#
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader


# ------------------------------------------------------------
# 2. SELECT CPU OR GPU
# ------------------------------------------------------------

# Check whether an NVIDIA GPU is available.
# If available, use CUDA.
# Otherwise, use CPU.

vin_device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", vin_device)

# Set random seed for reproducible results.
torch.manual_seed(42)


# ------------------------------------------------------------
# 3. LOAD OR CREATE REAL DATA
# ------------------------------------------------------------

# Select the data source.
#
# Options:
# "synthetic" -> Create sinusoidal data.
# "csv"       -> Load data from CSV.
# "npy"       -> Load data from NumPy NPY.
# "npz"       -> Load data from NumPy NPZ.
#
# We are using synthetic data for this practical.

vin_data_source = "synthetic"


# ------------------------------------------------------------
# OPTION A: SYNTHETIC DATA
# ------------------------------------------------------------

if vin_data_source == "synthetic":

    # Generate 3000 random values between -pi and pi.
    vin_t = torch.rand(3000) * 2 * np.pi - np.pi

    # Generate the x coordinates.
    vin_x = vin_t / np.pi

    # Generate the y coordinates using sine function.
    vin_y = torch.sin(vin_t)

    # Combine x and y into a 2D dataset.
    # Each row represents one sample: [x, y].

    vin_array = torch.stack(
        (vin_x, vin_y),
        dim=1
    ).numpy()


# ------------------------------------------------------------
# OPTION B: LOAD DATA FROM CSV
# ------------------------------------------------------------

elif vin_data_source == "csv":

    import pandas as pd

    # Read CSV file.
    vin_df = pd.read_csv("real_data.csv")

    # Select two numerical columns.
    # Change column names according to your dataset.

    vin_array = vin_df[["x", "y"]].values


# ------------------------------------------------------------
# OPTION C: LOAD DATA FROM NPY
# ------------------------------------------------------------

elif vin_data_source == "npy":

    # NPY contains a NumPy array.
    vin_array = np.load("real_data.npy")


# ------------------------------------------------------------
# OPTION D: LOAD DATA FROM NPZ
# ------------------------------------------------------------

elif vin_data_source == "npz":

    # NPZ can contain multiple arrays.
    # Here we assume the required array is stored under "X".

    vin_file = np.load("real_data.npz")

    vin_array = vin_file["X"]


# ------------------------------------------------------------
# 4. CONVERT DATA INTO PYTORCH TENSOR
# ------------------------------------------------------------

# Convert the loaded NumPy data into a PyTorch tensor.
# Neural networks generally use float32 values.

vin_real_original = torch.tensor(
    vin_array,
    dtype=torch.float32
)

# Check the shape of the data.
print("Real data shape:", vin_real_original.shape)

# Each row is a sample.
# Each column is a feature.
#
# Example:
# (3000, 2)
# Means 3000 samples and 2 features.

# Ensure there are at least two numerical features
# because we will visualize the first two features.

if vin_real_original.ndim != 2:
    raise ValueError("Data must be a 2D array.")

if vin_real_original.shape[0] < 2:
    raise ValueError("At least two samples are required.")

if vin_real_original.shape[1] < 2:
    raise ValueError("At least two features are required.")

if not torch.isfinite(vin_real_original).all():
    raise ValueError("Data contains NaN or infinite values.")


# ------------------------------------------------------------
# 5. NORMALIZE THE DATA
# ------------------------------------------------------------

# The Generator uses Tanh as its final activation.
# Tanh produces values between -1 and +1.
#
# Therefore, normalize real data to the same range.
#
# Formula:
#
# normalized = 2 * (x - min) / (max - min) - 1

vin_min_values = torch.min(
    vin_real_original,
    dim=0
).values

vin_max_values = torch.max(
    vin_real_original,
    dim=0
).values

vin_range = vin_max_values - vin_min_values

# Avoid division by zero if a feature is constant.
vin_safe_range = torch.where(
    vin_range == 0,
    torch.ones_like(vin_range),
    vin_range
)

vin_real_data = 2 * (
    (vin_real_original - vin_min_values) / vin_safe_range
) - 1

print("Normalized data shape:", vin_real_data.shape)


# ------------------------------------------------------------
# 6. CREATE DATASET AND DATALOADER
# ------------------------------------------------------------

# TensorDataset stores the real samples.
# No class labels are required for a basic GAN.

vin_dataset = TensorDataset(vin_real_data)

# DataLoader divides the dataset into batches.
# shuffle=True randomizes sample order in every epoch.

vin_loader = DataLoader(
    vin_dataset,
    batch_size=128,
    shuffle=True
)

print("Number of batches:", len(vin_loader))


# ------------------------------------------------------------
# 7. DEFINE THE GENERATOR
# ------------------------------------------------------------

# Generator:
#
# Input  : Random noise vector of size 16.
# Hidden : Two fully connected layers.
# Output : Fake sample having the same number of
#          features as the real data.
#
# ReLU introduces non-linearity.
# Tanh restricts output to [-1, +1].

class Generator(nn.Module):

    def __init__(self, vin_noise_size, vin_data_size):

        super().__init__()

        self.network = nn.Sequential(

            # Random noise -> 64 neurons
            nn.Linear(vin_noise_size, 64),

            # Activation function
            nn.ReLU(),

            # 64 -> 64 neurons
            nn.Linear(64, 64),

            nn.ReLU(),

            # 64 -> number of real-data features
            nn.Linear(64, vin_data_size),

            # Output between -1 and +1
            nn.Tanh()
        )

    def forward(self, vin_noise):

        # Pass the noise through the network.
        vin_output = self.network(vin_noise)

        return vin_output


# ------------------------------------------------------------
# 8. DEFINE THE DISCRIMINATOR
# ------------------------------------------------------------

# Discriminator:
#
# Input  : Real or fake numerical samples.
# Output : One score for each sample.
#
# A higher score indicates that the sample is more likely
# to be real.
#
# We use LeakyReLU to allow a small gradient even when
# the input to the activation is negative.
#
# IMPORTANT:
# There is no final Sigmoid here because we use
# BCEWithLogitsLoss, which combines Sigmoid and BCE.

class Discriminator(nn.Module):

    def __init__(self, vin_data_size):

        super().__init__()

        self.network = nn.Sequential(

            # Input features -> 64 neurons
            nn.Linear(vin_data_size, 64),

            # LeakyReLU activation
            nn.LeakyReLU(0.2),

            # 64 -> 64 neurons
            nn.Linear(64, 64),

            nn.LeakyReLU(0.2),

            # One output score per sample
            nn.Linear(64, 1)
        )

    def forward(self, vin_data):

        vin_output = self.network(vin_data)

        return vin_output


# ------------------------------------------------------------
# 9. INITIALIZE BOTH NETWORKS
# ------------------------------------------------------------

# Determine number of features in the real data.
vin_data_size = vin_real_data.shape[1]

# Dimension of random noise given to the Generator.
vin_noise_size = 16

# Create Generator and move it to CPU/GPU.
vin_generator = Generator(
    vin_noise_size,
    vin_data_size
).to(vin_device)

# Create Discriminator and move it to CPU/GPU.
vin_discriminator = Discriminator(
    vin_data_size
).to(vin_device)

print(vin_generator)
print(vin_discriminator)


# ------------------------------------------------------------
# 10. DEFINE LOSS FUNCTION
# ------------------------------------------------------------

# Binary Cross Entropy with Logits.
#
# The Discriminator learns:
# Real data -> target 1
# Fake data -> target 0
#
# During Generator training:
# Generated fake data -> target 1
#
# This encourages the Generator to fool the Discriminator.

vin_loss_function = nn.BCEWithLogitsLoss()


# ------------------------------------------------------------
# 11. DEFINE OPTIMIZERS
# ------------------------------------------------------------

# Each neural network has its own optimizer.
#
# Generator optimizer updates Generator parameters.
# Discriminator optimizer updates Discriminator parameters.
#
# Adam is an adaptive optimization algorithm.

vin_optimizer_G = optim.Adam(
    vin_generator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

vin_optimizer_D = optim.Adam(
    vin_discriminator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)


# ------------------------------------------------------------
# 12. TRAINING PARAMETERS
# ------------------------------------------------------------

vin_epochs = 200

# Store losses to visualize the training process.
vin_G_losses = []
vin_D_losses = []


# ------------------------------------------------------------
# 13. TRAIN THE GAN
# ------------------------------------------------------------

for vin_epoch in range(vin_epochs):

    # Reset total losses for this epoch.
    vin_total_G_loss = 0
    vin_total_D_loss = 0

    # Process one batch at a time.
    for vin_batch in vin_loader:

        # Extract real samples from the batch.
        #
        # TensorDataset returns a tuple.
        # The first element contains our real samples.

        vin_real = vin_batch[0].to(vin_device)

        # Find the current batch size.
        # This also handles the final smaller batch.

        vin_batch_size = vin_real.size(0)


        # ====================================================
        # STEP A: TRAIN THE DISCRIMINATOR
        # ====================================================

        # Enable gradients for Discriminator parameters.
        vin_discriminator.requires_grad_(True)

        # --------------------------------------------
        # A1. Generate fake data
        # --------------------------------------------

        # Create random noise from a normal distribution.
        #
        # Shape:
        # (batch size, noise dimension)
        #
        # Example:
        # (128, 16)

        vin_noise = torch.randn(
            vin_batch_size,
            vin_noise_size,
            device=vin_device
        )

        # Pass random noise through the Generator.
        # This produces fake samples.

        vin_fake = vin_generator(vin_noise)

        # detach() prevents gradients from flowing
        # back into the Generator during this step.
        #
        # We are updating only the Discriminator now.

        vin_fake = vin_fake.detach()


        # --------------------------------------------
        # A2. Discriminator predicts real samples
        # --------------------------------------------

        vin_real_output = vin_discriminator(vin_real)

        # The correct target for real data is 1.

        vin_real_labels = torch.ones_like(
            vin_real_output
        )


        # --------------------------------------------
        # A3. Discriminator predicts fake samples
        # --------------------------------------------

        vin_fake_output = vin_discriminator(vin_fake)

        # The correct target for fake data is 0.

        vin_fake_labels = torch.zeros_like(
            vin_fake_output
        )


        # --------------------------------------------
        # A4. Calculate Discriminator loss
        # --------------------------------------------

        # Loss for classifying real samples correctly.

        vin_loss_real = vin_loss_function(
            vin_real_output,
            vin_real_labels
        )

        # Loss for classifying fake samples correctly.

        vin_loss_fake = vin_loss_function(
            vin_fake_output,
            vin_fake_labels
        )

        # Total Discriminator loss.

        vin_D_loss = vin_loss_real + vin_loss_fake


        # --------------------------------------------
        # A5. Update Discriminator
        # --------------------------------------------

        # Clear old gradients.

        vin_optimizer_D.zero_grad()

        # Backpropagation calculates gradients.

        vin_D_loss.backward()

        # Update Discriminator parameters.

        vin_optimizer_D.step()


        # ====================================================
        # STEP B: TRAIN THE GENERATOR
        # ====================================================

        # Freeze Discriminator parameters.
        #
        # We still need to calculate gradients through
        # the Discriminator's operations to the Generator,
        # but we do not need to update Discriminator weights.

        vin_discriminator.requires_grad_(False)


        # --------------------------------------------
        # B1. Generate a fresh batch of fake data
        # --------------------------------------------

        vin_noise = torch.randn(
            vin_batch_size,
            vin_noise_size,
            device=vin_device
        )

        vin_fake = vin_generator(vin_noise)


        # --------------------------------------------
        # B2. Pass generated data through Discriminator
        # --------------------------------------------

        vin_fake_output = vin_discriminator(vin_fake)


        # --------------------------------------------
        # B3. Calculate Generator loss
        # --------------------------------------------

        # The Generator wants the Discriminator to
        # classify its fake samples as REAL.
        #
        # Therefore, the target is 1, not 0.

        vin_target_labels = torch.ones_like(
            vin_fake_output
        )

        vin_G_loss = vin_loss_function(
            vin_fake_output,
            vin_target_labels
        )


        # --------------------------------------------
        # B4. Update Generator
        # --------------------------------------------

        # Clear old Generator gradients.

        vin_optimizer_G.zero_grad()

        # Backpropagation.
        #
        # Gradients pass through the Discriminator
        # and update the Generator.

        vin_G_loss.backward()

        # Update Generator parameters.

        vin_optimizer_G.step()

        # Re-enable Discriminator gradients for
        # the next Discriminator training step.

        vin_discriminator.requires_grad_(True)


        # --------------------------------------------
        # B5. Add losses for monitoring
        # --------------------------------------------

        vin_total_D_loss += vin_D_loss.item()

        vin_total_G_loss += vin_G_loss.item()


    # --------------------------------------------------------
    # 14. PRINT EPOCH LOSSES
    # --------------------------------------------------------

    # Calculate average loss across all batches.

    vin_average_D_loss = vin_total_D_loss / len(vin_loader)

    vin_average_G_loss = vin_total_G_loss / len(vin_loader)

    vin_D_losses.append(vin_average_D_loss)

    vin_G_losses.append(vin_average_G_loss)

    # Print every 20 epochs.

    if (vin_epoch + 1) % 20 == 0:

        print(
            "Epoch:", vin_epoch + 1,
            "Discriminator Loss:", round(vin_average_D_loss, 4),
            "Generator Loss:", round(vin_average_G_loss, 4)
        )


# ------------------------------------------------------------
# 15. GENERATE NEW DATA
# ------------------------------------------------------------

# The Generator has now been trained.
#
# We can give it new random noise and generate
# samples that it has never seen before.

vin_generator.eval()

with torch.no_grad():

    # Create 1000 new random noise vectors.

    vin_noise = torch.randn(
        1000,
        vin_noise_size,
        device=vin_device
    )

    # Generate fake samples.

    vin_generated_data = vin_generator(
        vin_noise
    )

    # Move generated samples to CPU.
    vin_generated_data = vin_generated_data.cpu()


# ------------------------------------------------------------
# 16. CONVERT GENERATED DATA BACK TO ORIGINAL SCALE
# ------------------------------------------------------------

# The Generator produces normalized values in [-1, +1].
#
# Reverse the normalization so that generated samples
# can be compared with the original real data.

vin_generated_original = (
    (vin_generated_data + 1) / 2
) * vin_safe_range + vin_min_values

vin_generated_original = vin_generated_original.numpy()


# ------------------------------------------------------------
# 17. VISUALIZE REAL AND GENERATED DATA
# ------------------------------------------------------------

# Convert real data to NumPy for plotting.

vin_real_numpy = vin_real_original.numpy()

# Plot both real and generated samples.

plt.figure(figsize=(10, 6))

plt.scatter(
    vin_real_numpy[:, 0],
    vin_real_numpy[:, 1],
    label="Real Data",
    alpha=0.5
)

plt.scatter(
    vin_generated_original[:, 0],
    vin_generated_original[:, 1],
    label="Generated Data",
    alpha=0.5
)

plt.xlabel("X")
plt.ylabel("Y")

plt.title("GAN: Real Data vs Generated Data")

plt.legend()

plt.show()


# ------------------------------------------------------------
# 18. VISUALIZE TRAINING LOSSES
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    vin_D_losses,
    label="Discriminator Loss"
)

plt.plot(
    vin_G_losses,
    label="Generator Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("GAN Training Loss")

plt.legend()

plt.show()


# ------------------------------------------------------------
# 19. SAVE TRAINED MODELS
# ------------------------------------------------------------

# Save only the learned model parameters.
# These can be loaded later for generating new samples.

torch.save(
    vin_generator.state_dict(),
    "generator.pth"
)

torch.save(
    vin_discriminator.state_dict(),
    "discriminator.pth"
)

print("Generator saved successfully.")
print("Discriminator saved successfully.")

# Save generated samples as a NumPy file.

np.save(
    "generated_samples.npy",
    vin_generated_original
)

print("Generated samples saved successfully.")


# ============================================================
# END OF GAN PRACTICAL
# ============================================================

