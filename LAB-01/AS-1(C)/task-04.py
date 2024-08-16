import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_images(original, equalized, title1, title2):
    """Helper function to plot original and equalized images."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray', vmin=0, vmax=255)
    plt.title(title1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(equalized, cmap='gray', vmin=0, vmax=255)
    plt.title(title2)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Load a medical image (e.g., X-ray or MRI)
image_path = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\mri.jpeg"
medical_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded correctly
if medical_image is None:
    raise FileNotFoundError(f"Image not found at the path: {image_path}")

# Apply histogram equalization
equalized_medical_image = cv2.equalizeHist(medical_image)

# Plot original and equalized images
plot_images(medical_image, equalized_medical_image, 'Original Medical Image', 'Equalized Medical Image')


# Load a satellite image
image_path = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\satellite.jpeg"
satellite_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded correctly
if satellite_image is None:
    raise FileNotFoundError(f"Image not found at the path: {image_path}")

# Apply histogram equalization
equalized_satellite_image = cv2.equalizeHist(satellite_image)

# Plot original and equalized images
plot_images(satellite_image, equalized_satellite_image, 'Original Satellite Image', 'Equalized Satellite Image')


# Load a scanned document image
image_path = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\scanned.jpeg"
document_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded correctly
if document_image is None:
    raise FileNotFoundError(f"Image not found at the path: {image_path}")

# Apply histogram equalization
equalized_document_image = cv2.equalizeHist(document_image)

# Plot original and equalized images
plot_images(document_image, equalized_document_image, 'Original Document Image', 'Equalized Document Image')


# Load a night vision image
image_path = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\nv.jpeg"
night_vision_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded correctly
if night_vision_image is None:
    raise FileNotFoundError(f"Image not found at the path: {image_path}")

# Apply histogram equalization
equalized_night_vision_image = cv2.equalizeHist(night_vision_image)

# Plot original and equalized images
plot_images(night_vision_image, equalized_night_vision_image, 'Original Night Vision Image', 'Equalized Night Vision Image')



