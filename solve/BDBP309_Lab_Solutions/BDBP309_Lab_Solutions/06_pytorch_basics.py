"""Lab 6: PyTorch tensors, Dataset, DataLoader, transforms, autograd and save/load."""
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# ---------- Tensors
vin_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
vin_y = torch.tensor([[1.0], [0.0]])
print("Tensor:\n", vin_x)
print("Shape:", vin_x.shape)
print("Reshaped:\n", vin_x.reshape(4, 1))

# ---------- Dataset and DataLoader from tensors
vin_dataset = TensorDataset(vin_x, vin_y)
vin_loader = DataLoader(vin_dataset, batch_size=1, shuffle=True)
for vin_batch_x, vin_batch_y in vin_loader:
    print("Batch X:", vin_batch_x, "Batch y:", vin_batch_y)

# ---------- Custom Dataset class
class VinCustomDataset(Dataset):
    def __init__(self, vin_features, vin_labels):
        self.vin_features = vin_features
        self.vin_labels = vin_labels

    def __len__(self):
        return len(self.vin_features)

    def __getitem__(self, vin_index):
        return self.vin_features[vin_index], self.vin_labels[vin_index]

vin_custom_dataset = VinCustomDataset(vin_x, vin_y)
vin_custom_loader = DataLoader(vin_custom_dataset, batch_size=2, shuffle=True)

# ---------- Autograd
vin_parameter = torch.tensor(2.0, requires_grad=True)
vin_result = vin_parameter ** 2 + 3 * vin_parameter
vin_result.backward()
print("Gradient:", vin_parameter.grad)

# ---------- Model, loss, optimizer
vin_model = torch.nn.Linear(2, 1)
vin_optimizer = torch.optim.SGD(vin_model.parameters(), lr=0.01)
vin_loss_function = torch.nn.MSELoss()
for vin_batch_x, vin_batch_y in vin_custom_loader:
    vin_prediction = vin_model(vin_batch_x)
    vin_loss = vin_loss_function(vin_prediction, vin_batch_y)
    vin_optimizer.zero_grad()
    vin_loss.backward()
    vin_optimizer.step()

# ---------- Save and load model parameters
# This saves only model weights; recreate the same model architecture before loading.
torch.save(vin_model.state_dict(), "06_pytorch_model.pth")
vin_loaded_model = torch.nn.Linear(2, 1)
vin_loaded_model.load_state_dict(torch.load("06_pytorch_model.pth", map_location="cpu", weights_only=True))
vin_loaded_model.eval()
print("Loaded model prediction:", vin_loaded_model(vin_x))

# ---------- Optional image transform example (requires torchvision)
# from torchvision import transforms
# vin_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
