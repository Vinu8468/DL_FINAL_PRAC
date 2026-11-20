
#------------
# 9. Combine tensors

vin_a = torch.tensor([1, 2, 3])
vin_b = torch.tensor([4, 5, 6])

# Concatenate along an existing dimension
vin_concatenated = torch.cat((vin_a, vin_b), dim=0)

# Stack tensors along a new dimension
vin_stacked = torch.stack((vin_a, vin_b), dim=0)

print(vin_concatenated)
print(vin_stacked)


#------------
# 10. Convert NumPy arrays to PyTorch tensors and back

vin_numpy = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

# Convert NumPy array to tensor
vin_tensor = torch.from_numpy(vin_numpy)

# Alternative conversion
vin_tensor_2 = torch.tensor(vin_numpy, dtype=torch.float32)

# Convert tensor to NumPy array (tensor must be on CPU)
vin_numpy_again = vin_tensor.cpu().numpy()

print(vin_tensor)
print(vin_numpy_again)


#------------
# 11. Convert Pandas data to PyTorch tensors

# Example DataFrame; replace this with pd.read_csv("data.csv") when needed
vin_df = pd.DataFrame({
    "Feature1": [1.0, 2.0, 3.0, 4.0],
    "Feature2": [5.0, 6.0, 7.0, 8.0],
    "Label": [0, 1, 0, 1]
})

# Separate features (X) and target (y)
vin_X = vin_df.drop("Label", axis=1).to_numpy(dtype=np.float32)
vin_y = vin_df["Label"].to_numpy(dtype=np.int64)

# Convert features and labels to tensors
vin_X_tensor = torch.tensor(vin_X, dtype=torch.float32)
vin_y_tensor = torch.tensor(vin_y, dtype=torch.long)

print(vin_X_tensor.shape)
print(vin_y_tensor.shape)


#------------
# 12. Save and load PyTorch tensors (.pt and .pth)

vin_tensor_to_save = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Save a tensor
# torch.save(vin_tensor_to_save, "tensor.pt")

# Load a tensor saved with torch.save
# vin_loaded_tensor = torch.load("tensor.pt", map_location="cpu", weights_only=True)

# Save multiple tensors in a dictionary
vin_data_to_save = {
    "X": vin_X_tensor,
    "y": vin_y_tensor
}

# Save the dictionary
# torch.save(vin_data_to_save, "dataset.pth")

# Load the dictionary
# vin_loaded_data = torch.load("dataset.pth", map_location="cpu", weights_only=True)
# vin_loaded_X = vin_loaded_data["X"]
# vin_loaded_y = vin_loaded_data["y"]

# .pt and .pth are commonly used extensions; both use torch.save() and torch.load().
# The extension alone does not tell you whether the file contains a tensor,
# a dictionary of tensors, or a model state dictionary.


#------------
# 13. Move tensors between CPU and GPU

# Select GPU when available; otherwise use CPU
vin_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vin_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Move tensor to selected device
vin_device_tensor = vin_tensor.to(vin_device)

print("Selected device:", vin_device)
print("Tensor device:", vin_device_tensor.device)

# Move tensor back to CPU
vin_cpu_tensor = vin_device_tensor.cpu()


#------------
# 14. Detach tensor and convert to NumPy

vin_tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
vin_result = vin_tensor * 2

# Detach removes the tensor from the gradient-tracking computation graph
vin_detached = vin_result.detach()

# Convert detached CPU tensor to NumPy
vin_result_numpy = vin_detached.cpu().numpy()

print(vin_result_numpy)


#------------
# 15. Important notes for deep learning

# For most neural network inputs and continuous features, use float32.
# For class labels used by nn.CrossEntropyLoss, use long (int64).
# For binary labels used by nn.BCELoss, use float32 with values 0 and 1.
# A common image batch shape is (batch_size, channels, height, width).
# A common tabular batch shape is (batch_size, number_of_features).
# A common sequence batch shape is (batch_size, sequence_length).
# Keep tensors on the same device before performing operations on them.

