
# 6. ACCESS A PARTICULAR SAMPLE OR BATCH
#------------

# Get one sample directly from a Dataset.
vin_sample_X, vin_sample_y = vin_dataset[0]
print("Single sample:", vin_sample_X, vin_sample_y)

# Get one batch from a DataLoader.
vin_batch_X, vin_batch_y = next(iter(vin_loader))
print("One batch:", vin_batch_X, vin_batch_y)

#------------
# 7. SUBSET OF A DATASET
#------------

# Select particular sample indices without copying the original dataset.
vin_selected_indices = [0, 2, 4]
vin_subset = Subset(vin_dataset, vin_selected_indices)
vin_subset_loader = DataLoader(vin_subset, batch_size=2, shuffle=False)

for vin_batch_X, vin_batch_y in vin_subset_loader:
    print("Subset batch:", vin_batch_X, vin_batch_y)

#------------
# 8. CUSTOM DATASET WITH TRANSFORM
#------------

class vin_TransformDataset(Dataset):
    def __init__(self, vin_X, vin_y, vin_transform=None):
        self.vin_X = vin_X
        self.vin_y = vin_y
        self.vin_transform = vin_transform

    def __len__(self):
        return len(self.vin_X)

    def __getitem__(self, vin_index):
        vin_sample_X = self.vin_X[vin_index]
        vin_sample_y = self.vin_y[vin_index]

        if self.vin_transform is not None:
            vin_sample_X = self.vin_transform(vin_sample_X)

        return vin_sample_X, vin_sample_y

# Example transform: normalize each feature vector by its maximum absolute value.
def vin_normalize(vin_x):
    vin_max = torch.max(torch.abs(vin_x))
    if vin_max > 0:
        vin_x = vin_x / vin_max
    return vin_x

vin_transform_dataset = vin_TransformDataset(
    vin_X,
    vin_y,
    vin_transform=vin_normalize
)

vin_transform_loader = DataLoader(
    vin_transform_dataset,
    batch_size=2,
    shuffle=True
)

#------------
# 9. DROP THE LAST INCOMPLETE BATCH
#------------

vin_loader_drop_last = DataLoader(
    vin_dataset,
    batch_size=2,
    shuffle=False,
    drop_last=True
)

for vin_batch_X, vin_batch_y in vin_loader_drop_last:
    print("Full batch:", vin_batch_X, vin_batch_y)

#------------
# 10. COMMON TRAINING LOOP USING A DATALOADER
#------------

# This is a basic example showing how batches are used during training.
# Replace the simple model with the architecture required by the question.

vin_model = torch.nn.Linear(2, 2)
vin_criterion = torch.nn.CrossEntropyLoss()
vin_optimizer = torch.optim.Adam(vin_model.parameters(), lr=0.001)

for vin_epoch in range(2):
    vin_total_loss = 0.0

    for vin_batch_X, vin_batch_y in vin_train_loader:
        vin_output = vin_model(vin_batch_X)
        vin_loss = vin_criterion(vin_output, vin_batch_y)

        vin_optimizer.zero_grad()
        vin_loss.backward()
        vin_optimizer.step()

        vin_total_loss = vin_total_loss + vin_loss.item()

    print("Epoch:", vin_epoch + 1, "Loss:", vin_total_loss)

#------------
# 11. IMPORTANT NOTES
#------------

# Dataset:
#   Stores/accesses individual samples.
#   Must define __len__() and __getitem__() in a custom map-style Dataset.
#
# DataLoader:
#   Creates batches and can shuffle the samples.
#   Used to loop through data during training and testing.
#
# For classification:
#   X is usually torch.float32.
#   Class labels y are usually torch.long when using CrossEntropyLoss.
#
# For binary classification with BCELoss:
#   Model output and y should usually both have floating-point values.
#
# For testing/validation:
#   Usually set shuffle=False.

0
