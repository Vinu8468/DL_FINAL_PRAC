
print("Start epoch:", vin_start_epoch)
print("Previous loss:", vin_previous_loss)

#------------ 7. Move model to CPU or GPU

vin_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vin_model = vin_model.to(vin_device)

#------------ 8. Inference using loaded model

vin_x = torch.randn(4, 10).to(vin_device)

with torch.no_grad():
    vin_output = vin_model(vin_x)

print(vin_output)

#------------ 9. Save and load a tensor

vin_tensor = torch.tensor([1, 2, 3, 4])

torch.save(vin_tensor, "tensor.pt")
vin_loaded_tensor = torch.load("tensor.pt", map_location="cpu", weights_only=True)

print(vin_loaded_tensor)

#------------ 10. Save multiple items in one file

vin_data = {
    "X": torch.randn(100, 10),
    "y": torch.randint(0, 2, (100,))
}

torch.save(vin_data, "data.pt")
vin_loaded_data = torch.load("data.pt", map_location="cpu", weights_only=True)

vin_X = vin_loaded_data["X"]
vin_y = vin_loaded_data["y"]

print(vin_X.shape)
print(vin_y.shape)

#------------ 11. Load pretrained ResNet-18 (optional; internet may be needed first time)

# import torchvision.models as models
# vin_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# vin_resnet.eval()

