import cv2
import numpy as np
import matplotlib.pyplot as plt
image_path =  "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\gray.jpeg" 
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Could not load image.")
else:
    threshold_value = 127
    _, segmented_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Segmented Image')
    plt.imshow(segmented_image, cmap='gray')
    plt.axis('off')

    plt.show()
