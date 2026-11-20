"""Lab 11: 2D convolution and max-pooling implemented with loops."""
import numpy as np

# ---------- Make a 32 x 32 test image and a 3 x 3 kernel
vin_image = np.random.rand(32, 32)
vin_kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=float)

def vin_convolution(vin_image, vin_kernel):
    vin_image_height = vin_image.shape[0]
    vin_image_width = vin_image.shape[1]
    vin_kernel_height = vin_kernel.shape[0]
    vin_kernel_width = vin_kernel.shape[1]
    vin_output_height = vin_image_height - vin_kernel_height + 1
    vin_output_width = vin_image_width - vin_kernel_width + 1
    vin_output = np.zeros((vin_output_height, vin_output_width))
    for vin_i in range(vin_output_height):
        for vin_j in range(vin_output_width):
            vin_region = vin_image[vin_i:vin_i + vin_kernel_height, vin_j:vin_j + vin_kernel_width]
            vin_output[vin_i, vin_j] = np.sum(vin_region * vin_kernel)
    return vin_output

def vin_max_pool(vin_image, vin_pool_size=2):
    vin_height = vin_image.shape[0]
    vin_width = vin_image.shape[1]
    vin_out_height = vin_height // vin_pool_size
    vin_out_width = vin_width // vin_pool_size
    vin_output = np.zeros((vin_out_height, vin_out_width))
    for vin_i in range(vin_out_height):
        for vin_j in range(vin_out_width):
            vin_region = vin_image[
                vin_i * vin_pool_size:(vin_i + 1) * vin_pool_size,
                vin_j * vin_pool_size:(vin_j + 1) * vin_pool_size
            ]
            vin_output[vin_i, vin_j] = np.max(vin_region)
    return vin_output

vin_conv_output = vin_convolution(vin_image, vin_kernel)
vin_pool_output = vin_max_pool(vin_conv_output, 2)
print("Input image:", vin_image.shape)
print("After 3x3 valid convolution:", vin_conv_output.shape)
print("After 2x2 max pooling:", vin_pool_output.shape)
