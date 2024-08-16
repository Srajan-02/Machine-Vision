import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

image_path = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\gray.jpeg"  
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

gamma_values = [0.5, 1.0, 2.0]
gamma_corrected_images = [gamma_correction(original_image, gamma) for gamma in gamma_values]

plt.figure(figsize=(15, 5))

# Displaying original image
plt.subplot(1, 4, 1)
plt.title('Original Image')
plt.imshow(original_image, cmap='gray')
plt.axis('off')

for i, gamma in enumerate(gamma_values):
    plt.subplot(1, 4, i + 2)
    plt.title(f'Gamma = {gamma}')
    plt.imshow(gamma_corrected_images[i], cmap='gray')
    plt.axis('off')

plt.show()
