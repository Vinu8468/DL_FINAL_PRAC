

vin_normalize = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

vin_normalized = vin_normalize(vin_image_tensor)

#------------ 5. Resize + tensor + normalize (pipeline)

vin_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

vin_processed_image = vin_transform(vin_image)

#------------ 6. Random horizontal flip

vin_flip = transforms.RandomHorizontalFlip(p=0.5)
vin_flipped = vin_flip(vin_image)

#------------ 7. Random vertical flip

vin_vflip = transforms.RandomVerticalFlip(p=0.5)
vin_vertical_flipped = vin_vflip(vin_image)

#------------ 8. Random rotation

vin_rotation = transforms.RandomRotation(degrees=30)
vin_rotated = vin_rotation(vin_image)

#------------ 9. Random crop

vin_crop = transforms.RandomCrop((180, 180))
vin_cropped = vin_crop(vin_image)

#------------ 10. Random resized crop

vin_random_crop = transforms.RandomResizedCrop((224, 224))
vin_random_cropped = vin_random_crop(vin_image)

#------------ 11. Color jitter (RGB images)

vin_color = transforms.ColorJitter(
    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
)
vin_color_changed = vin_color(vin_image)

#------------ 12. Grayscale

vin_gray = transforms.Grayscale(num_output_channels=1)
vin_gray_image = vin_gray(vin_image)

#------------ 13. Display transformed images

vin_images = [vin_image, vin_image_resized, vin_flipped, vin_rotated, vin_cropped]
vin_titles = ["Original", "Resized", "Horizontal Flip", "Rotation", "Crop"]

plt.figure(figsize=(12, 4))

for vin_i in range(len(vin_images)):
    plt.subplot(1, 5, vin_i + 1)
    plt.imshow(vin_images[vin_i])
    plt.title(vin_titles[vin_i])
    plt.axis("off")

plt.show()

#------------ 14. ImageFolder dataset loading

vin_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

vin_dataset = ImageFolder("dataset/train", transform=vin_transform)

print(vin_dataset.classes)
print(vin_dataset.class_to_idx)
print(len(vin_dataset))

#------------ 15. DataLoader for images

vin_loader = DataLoader(vin_dataset, batch_size=32, shuffle=True)

for vin_images, vin_labels in vin_loader:
    print(vin_images.shape)  # B, C, H, W
    print(vin_labels.shape)
    break

#------------ 16. Training vs testing transforms

vin_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

vin_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

#------------ 17. Add batch dimension to a single image

vin_single_image = torch.randn(3, 224, 224)
vin_batch_image = vin_single_image.unsqueeze(0)

print(vin_batch_image.shape)  # 1, 3, 224, 224

#------------ 18. Convert tensor back to image (simple display)

vin_tensor_image = vin_image_tensor.permute(1, 2, 0).numpy()

plt.imshow(vin_tensor_image)
plt.axis("off")
plt.show()

0
