"""Lab 21 (additional): Fine-tune a medical image foundation model (BiomedCLIP).

Expected folder structure:
medical_images/
    class_1/ image1.png image2.png ...
    class_2/ image1.png image2.png ...

Install: pip install open_clip_torch torchvision
The first run downloads the pretrained BiomedCLIP checkpoint from Hugging Face.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets
import open_clip

vin_data_directory = "medical_images"
vin_model_identifier = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
vin_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vin_batch_size = 4
vin_epochs = 2

if not os.path.isdir(vin_data_directory):
    raise FileNotFoundError(
        "Create medical_images/class_name/image files first. "
        "Each class must be a subfolder containing its images."
    )

# ---------- Load BiomedCLIP and its image preprocessing transforms
vin_model, vin_train_transform, vin_validation_transform = open_clip.create_model_and_transforms(
    vin_model_identifier
)
vin_model = vin_model.to(vin_device)

# ---------- Load custom labelled medical image dataset
# A single ImageFolder is split here for a compact practical. For formal work, create
# separate train/validation/test folders and use a patient-level split to avoid leakage.
vin_full_dataset = datasets.ImageFolder(vin_data_directory, transform=vin_train_transform)
vin_validation_source = datasets.ImageFolder(vin_data_directory, transform=vin_validation_transform)
if len(vin_full_dataset.classes) < 2:
    raise ValueError("Provide at least two class subfolders for classification fine-tuning.")
vin_train_size = int(0.8 * len(vin_full_dataset))
vin_validation_size = len(vin_full_dataset) - vin_train_size
vin_train_subset, vin_validation_subset = random_split(
    range(len(vin_full_dataset)), [vin_train_size, vin_validation_size],
    generator=torch.Generator().manual_seed(42)
)
vin_train_dataset = Subset(vin_full_dataset, vin_train_subset.indices)
vin_validation_dataset = Subset(vin_validation_source, vin_validation_subset.indices)
vin_train_loader = DataLoader(vin_train_dataset, batch_size=vin_batch_size, shuffle=True)
vin_validation_loader = DataLoader(vin_validation_dataset, batch_size=vin_batch_size, shuffle=False)

# ---------- Fine-tune the visual tower and train a task-specific classification head
# Freeze text encoder; it is not required for the image classification head.
for vin_parameter in vin_model.parameters():
    vin_parameter.requires_grad = False
for vin_parameter in vin_model.visual.parameters():
    vin_parameter.requires_grad = True

vin_feature_dimension = vin_model.visual.output_dim
vin_classifier = nn.Linear(vin_feature_dimension, len(vin_full_dataset.classes)).to(vin_device)
vin_optimizer = torch.optim.Adam(
    list(vin_model.visual.parameters()) + list(vin_classifier.parameters()), lr=1e-5
)
vin_loss_function = nn.CrossEntropyLoss()

for vin_epoch in range(vin_epochs):
    vin_model.train()
    vin_classifier.train()
    vin_total_loss = 0.0
    for vin_images, vin_labels in vin_train_loader:
        vin_images = vin_images.to(vin_device)
        vin_labels = vin_labels.to(vin_device)
        vin_features = vin_model.encode_image(vin_images)
        vin_logits = vin_classifier(vin_features)
        vin_loss = vin_loss_function(vin_logits, vin_labels)
        vin_optimizer.zero_grad()
        vin_loss.backward()
        vin_optimizer.step()
        vin_total_loss = vin_total_loss + vin_loss.item()
    print("Epoch", vin_epoch + 1, "training loss", vin_total_loss / max(1, len(vin_train_loader)))

vin_model.eval()
vin_classifier.eval()
vin_correct = 0
vin_total = 0
with torch.no_grad():
    for vin_images, vin_labels in vin_validation_loader:
        vin_images = vin_images.to(vin_device)
        vin_labels = vin_labels.to(vin_device)
        vin_logits = vin_classifier(vin_model.encode_image(vin_images))
        vin_correct = vin_correct + (torch.argmax(vin_logits, dim=1) == vin_labels).sum().item()
        vin_total = vin_total + len(vin_labels)
if vin_total > 0:
    print("Validation accuracy:", vin_correct / vin_total)
else:
    print("Validation set is empty; add more images or adjust the split.")

torch.save({
    "visual_encoder": vin_model.visual.state_dict(),
    "classifier": vin_classifier.state_dict(),
    "class_names": vin_full_dataset.classes
}, "21_biomedclip_finetuned.pth")
print("Saved fine-tuned visual encoder and classifier.")
