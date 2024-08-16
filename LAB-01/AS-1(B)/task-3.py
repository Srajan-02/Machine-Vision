import cv2
import numpy as np
import matplotlib.pyplot as plt

def log_transform(image):
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(1 + image))
    log_image = np.array(log_image, dtype=np.uint8)
    
    return log_image

# Load a grayscale image
image_path = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\gray.jpeg"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

log_transformed_image = log_transform(original_image)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(original_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Log Transformed Image')
plt.imshow(log_transformed_image, cmap='gray')
plt.axis('off')

plt.show()
