import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the color image
image_path = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\rgb.jpeg"
image = cv2.imread(image_path)

# Check if the image is loaded correctly
if image is None:
    raise FileNotFoundError(f"Image not found at the path: {image_path}")

# Convert the image from BGR (OpenCV default) to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Separate the RGB channels
red_channel, green_channel, blue_channel = cv2.split(image_rgb)

# Display the separated channels
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(red_channel, cmap='Reds')
plt.title('Red Channel')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(green_channel, cmap='Greens')
plt.title('Green Channel')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(blue_channel, cmap='Blues')
plt.title('Blue Channel')
plt.axis('off')

plt.tight_layout()
plt.show()


# Apply histogram equalization to each channel
red_equalized = cv2.equalizeHist(red_channel)
green_equalized = cv2.equalizeHist(green_channel)
blue_equalized = cv2.equalizeHist(blue_channel)

# Display equalized channels
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(red_equalized, cmap='Reds')
plt.title('Equalized Red Channel')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(green_equalized, cmap='Greens')
plt.title('Equalized Green Channel')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(blue_equalized, cmap='Blues')
plt.title('Equalized Blue Channel')
plt.axis('off')

plt.tight_layout()
plt.show()
