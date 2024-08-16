import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load a low contrast image
image_path = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\low_contrast.jpeg"
low_contrast_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded correctly
if low_contrast_image is None:
    raise FileNotFoundError(f"Image not found at the path: {image_path}")

# Display the original low contrast image
plt.figure(figsize=(5, 5))
plt.imshow(low_contrast_image, cmap='gray', vmin=0, vmax=255)
plt.title('Original Low Contrast Image')
plt.axis('off')
plt.show()


# Apply histogram equalization
equalized_image = cv2.equalizeHist(low_contrast_image)

# Display original and equalized images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(low_contrast_image, cmap='gray', vmin=0, vmax=255)
plt.title('Original Low Contrast Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap='gray', vmin=0, vmax=255)
plt.title('Equalized Image')
plt.axis('off')

plt.tight_layout()
plt.show()
