import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image in grayscale
image_path = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\gray.jpeg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded correctly
if image is None:
    raise FileNotFoundError(f"Image not found at the path: {image_path}")

# Perform standard histogram equalization
equalized_image = cv2.equalizeHist(image)

# Implement Adaptive Histogram Equalization (AHE)
def adaptive_histogram_equalization(img, tile_size):
    img = img.astype(np.float32)
    img_ahe = np.zeros_like(img)
    h, w = img.shape
    tile_h, tile_w = tile_size

    for i in range(0, h, tile_h):
        for j in range(0, w, tile_w):
            tile = img[i:i+tile_h, j:j+tile_w]
            tile_equalized = cv2.equalizeHist(tile.astype(np.uint8))
            img_ahe[i:i+tile_h, j:j+tile_w] = tile_equalized

    return img_ahe.astype(np.uint8)

# Define tile size for AHE
tile_size = (32, 32)
ahe_image = adaptive_histogram_equalization(image, tile_size)

# Display original, standard histogram equalized, and AHE images
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(equalized_image, cmap='gray', vmin=0, vmax=255)
plt.title('Standard Histogram Equalization')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(ahe_image, cmap='gray', vmin=0, vmax=255)
plt.title('Adaptive Histogram Equalization (AHE)')
plt.axis('off')

plt.tight_layout()
plt.show()


# Implement Contrast Limited Adaptive Histogram Equalization (CLAHE)
def contrast_limited_ahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_image = clahe.apply(img)
    return clahe_image

# Apply CLAHE
clahe_image = contrast_limited_ahe(image)

# Display original, AHE, and CLAHE images
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(equalized_image, cmap='gray', vmin=0, vmax=255)
plt.title('Standard Histogram Equalization')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(ahe_image, cmap='gray', vmin=0, vmax=255)
plt.title('AHE Image')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(clahe_image, cmap='gray', vmin=0, vmax=255)
plt.title('CLAHE Image')
plt.axis('off')

plt.tight_layout()
plt.show()


