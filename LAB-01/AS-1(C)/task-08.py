import cv2
import matplotlib.pyplot as plt

# Load the original image
image_path = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\gray.jpeg"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded correctly
if original_image is None:
    raise FileNotFoundError(f"Image not found at the path: {image_path}")

# Perform Otsu's thresholding for segmentation
_, segmented_original = cv2.threshold(original_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the original image and its segmentation result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_original, cmap='gray', vmin=0, vmax=255)
plt.title('Segmented Original Image')
plt.axis('off')

plt.tight_layout()
plt.show()


# Apply histogram equalization to the original image
equalized_image = cv2.equalizeHist(original_image)

# Perform Otsu's thresholding for segmentation on the equalized image
_, segmented_equalized = cv2.threshold(equalized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the equalized image and its segmentation result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(equalized_image, cmap='gray', vmin=0, vmax=255)
plt.title('Equalized Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_equalized, cmap='gray', vmin=0, vmax=255)
plt.title('Segmented Equalized Image')
plt.axis('off')

plt.tight_layout()
plt.show()


