import cv2
import matplotlib.pyplot as plt

# Load a grayscale image
image_path = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\gray.jpeg"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Applying the image negative transformation
negative_image = 255 - original_image

# Displaying the original and the negative image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(original_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Negative Image')
plt.imshow(negative_image, cmap='gray')
plt.axis('off')

plt.show()
