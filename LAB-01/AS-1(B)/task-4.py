import cv2
import numpy as np
import matplotlib.pyplot as plt

def image_negative(image):
    # Apply the image negative transformation
    return 255 - image

def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(image, table)

def log_transform(image):
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(1 + image))
    log_image = np.array(log_image, dtype=np.uint8)
    
    return log_image

image_path = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\gray.jpeg"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply transformations
negative_image = image_negative(original_image)
gamma_corrected_image = gamma_correction(original_image, 2.0)
log_transformed_image = log_transform(original_image)

plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.title('Original Image')
plt.imshow(original_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Negative Image')
plt.imshow(negative_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('Gamma Corrected (γ=2.0)')
plt.imshow(gamma_corrected_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('Log Transformed Image')
plt.imshow(log_transformed_image, cmap='gray')
plt.axis('off')

plt.show()
