
if transforms is not None:
    vin_train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    if os.path.exists(vin_image_path):
        vin_image = Image.open(vin_image_path).convert("RGB")
        vin_augmented = vin_train_transform(vin_image)
        print("Augmented tensor shape:", vin_augmented.shape)

#------------
# 15. LOAD IMAGE DATASET FROM FOLDERS USING IMAGEFOLDER

# Expected folder structure:
# dataset/
#     cat/
#         image1.jpg
#         image2.jpg
#     dog/
#         image1.jpg
#         image2.jpg

if datasets is not None and transforms is not None:
    vin_folder_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    vin_dataset_path = "dataset"

    if os.path.isdir(vin_dataset_path):
        vin_dataset = datasets.ImageFolder(
            root=vin_dataset_path,
            transform=vin_folder_transform
        )

        print("Number of images:", len(vin_dataset))
        print("Class names:", vin_dataset.classes)
        print("Class-to-index mapping:", vin_dataset.class_to_idx)

        # Get one image and its label
        vin_image_tensor, vin_label = vin_dataset[0]
        print("Image tensor shape:", vin_image_tensor.shape)
        print("Label:", vin_label)

#------------
# 16. CREATE IMAGE DATALOADER

from torch.utils.data import DataLoader

if datasets is not None and transforms is not None:
    vin_dataset_path = "dataset"

    if os.path.isdir(vin_dataset_path):
        vin_dataset = datasets.ImageFolder(
            root=vin_dataset_path,
            transform=transforms.ToTensor()
        )

        vin_loader = DataLoader(
            vin_dataset,
            batch_size=32,
            shuffle=True
        )

        for vin_images, vin_labels in vin_loader:
            print("Batch image shape:", vin_images.shape)
            print("Batch label shape:", vin_labels.shape)
            break

#------------
# 17. SAVE PROCESSED IMAGE

if os.path.exists(vin_image_path):
    vin_image = Image.open(vin_image_path).convert("RGB")
    vin_resized = vin_image.resize((224, 224))

    vin_resized.save("resized_image.png")

    if cv2 is not None:
        vin_cv_image = cv2.imread(vin_image_path)
        vin_resized_cv = cv2.resize(vin_cv_image, (224, 224))
        cv2.imwrite("resized_image_cv.png", vin_resized_cv)

#------------
# 18. IMPORTANT IMAGE SHAPE REFERENCE

# PIL image size: (width, height)
# NumPy color image: (height, width, channels)
# NumPy grayscale image: (height, width)
# PyTorch color tensor: (channels, height, width)
# PyTorch batch: (batch, channels, height, width)

# RGB has 3 channels
# Grayscale has 1 channel
# OpenCV reads color images as BGR
# PIL reads color images as RGB
# ToTensor converts uint8 pixel values from 0-255 to float values from 0-1

