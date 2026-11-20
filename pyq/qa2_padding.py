import cv2
import numpy as np
import matplotlib.pyplot as plt

def cnn_padding(image_path, padding=2, displayimage=True):

    vin_img = cv2.imread(image_path, 0)

    vin_padded = np.pad(vin_img, padding, mode='constant', constant_values=0)

    print("Original size:", vin_img.shape)
    print("Padded size:", vin_padded.shape)

    if displayimage:
        plt.subplot(1, 2, 1)
        plt.imshow(vin_img, cmap='gray')
        plt.title("Original")

        plt.subplot(1, 2, 2)
        plt.imshow(vin_padded, cmap='gray')
        plt.title("Padded")

        plt.show()

    return vin_padded


vin_result = cnn_padding(
    r"C:\Users\vinee\OneDrive\Scans\Pictures\Screenshots\Screenshot 2026-11-12 114255.png",
    padding=2
)
