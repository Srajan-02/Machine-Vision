import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the distorted MRI image
distorted_mri = cv2.imread("C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\mri.jpeg", cv2.IMREAD_GRAYSCALE)
plt.imshow(distorted_mri, cmap='gray')
plt.title('Distorted MRI Image')
plt.show()


# Coordinates of key points in the distorted MRI image
distorted_points = np.array([[120, 80], [300, 100], [100, 250], [280, 270]])  


# Coordinates of corresponding points in the reference anatomical model
reference_points = np.array([[100, 100], [400, 100], [100, 400], [400, 400]]) 


# Compute the transformation matrix
transformation_matrix = cv2.getPerspectiveTransform(distorted_points.astype(np.float32), reference_points.astype(np.float32))

# Apply the perspective transformation
rectified_mri = cv2.warpPerspective(distorted_mri, transformation_matrix, (500, 500))  # Adjust the size as needed


# Apply transformation with bilinear interpolation
rectified_mri_bilinear = cv2.warpPerspective(distorted_mri, transformation_matrix, (500, 500), flags=cv2.INTER_LINEAR)

# Display the rectified MRI image and compare it with the reference model
plt.subplot(1, 2, 1)
plt.imshow(distorted_mri, cmap='gray')
plt.title('Distorted MRI Image')

plt.subplot(1, 2, 2)
plt.imshow(rectified_mri_bilinear, cmap='gray')
plt.title('Rectified MRI Image (Bilinear Interpolation)')
plt.show()
