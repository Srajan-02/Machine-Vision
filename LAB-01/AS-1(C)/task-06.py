import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the original image
image_path = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\gray.jpeg"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded correctly
if original_image is None:
    raise FileNotFoundError(f"Image not found at the path: {image_path}")

# Resize the image to different scales
scale_50 = cv2.resize(original_image, (0, 0), fx=0.5, fy=0.5)
scale_100 = original_image.copy()  # Original scale (100%)
scale_200 = cv2.resize(original_image, (0, 0), fx=2.0, fy=2.0)

# Display the resized images
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(scale_50, cmap='gray', vmin=0, vmax=255)
plt.title('50% Scale')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(scale_100, cmap='gray', vmin=0, vmax=255)
plt.title('100% Scale')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(scale_200, cmap='gray', vmin=0, vmax=255)
plt.title('200% Scale')
plt.axis('off')

plt.tight_layout()
plt.show()


# Apply histogram equalization to each scaled image
equalized_50 = cv2.equalizeHist(scale_50)
equalized_100 = cv2.equalizeHist(scale_100)
equalized_200 = cv2.equalizeHist(scale_200)

# Display the equalized images
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(equalized_50, cmap='gray', vmin=0, vmax=255)
plt.title('Equalized 50% Scale')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(equalized_100, cmap='gray', vmin=0, vmax=255)
plt.title('Equalized 100% Scale')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(equalized_200, cmap='gray', vmin=0, vmax=255)
plt.title('Equalized 200% Scale')
plt.axis('off')

plt.tight_layout()
plt.show()


