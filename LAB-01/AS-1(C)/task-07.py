import cv2
import matplotlib.pyplot as plt

# Load images with varying levels of detail and contrast
image_paths = ["C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\low_contrast.jpeg", "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\high_contrast.jpeg", "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\moderate_detail.jpeg"]  
images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths]

# Check if the images are loaded correctly
for i, image in enumerate(images):
    if image is None:
        raise FileNotFoundError(f"Image not found at the path: {image_paths[i]}")

# Display the original images
plt.figure(figsize=(15, 5))
for i, image in enumerate(images):
    plt.subplot(1, 3, i+1)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Original Image {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Apply histogram equalization
equalized_images = [cv2.equalizeHist(image) for image in images]

# Display the equalized images
plt.figure(figsize=(15, 5))
for i, equalized_image in enumerate(equalized_images):
    plt.subplot(1, 3, i+1)
    plt.imshow(equalized_image, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Equalized Image {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()
