import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the distorted architectural image
distorted_arch_image = cv2.imread("C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\architectural.jpeg")
plt.imshow(cv2.cvtColor(distorted_arch_image, cv2.COLOR_BGR2RGB))
plt.title('Distorted Architectural Image')
plt.show()

# Coordinates of key points on the distorted architectural image
distorted_points = np.array([[100, 150], [500, 150], [100, 400], [500, 400]])  

# Corresponding points in the rectified image
rectified_points = np.array([[100, 100], [500, 100], [100, 400], [500, 400]]) 

# Compute the perspective transformation matrix
transformation_matrix = cv2.getPerspectiveTransform(distorted_points.astype(np.float32), rectified_points.astype(np.float32))

# Apply the perspective transformation
rectified_image = cv2.warpPerspective(distorted_arch_image, transformation_matrix, (600, 600)) 

# Apply transformation using bilinear interpolation
rectified_image_bilinear = cv2.warpPerspective(distorted_arch_image, transformation_matrix, (600, 600), flags=cv2.INTER_LINEAR)

# Display the original and rectified images side by side
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(distorted_arch_image, cv2.COLOR_BGR2RGB))
plt.title('Original Distorted Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(rectified_image_bilinear, cv2.COLOR_BGR2RGB))
plt.title('Rectified Image (Bilinear Interpolation)')
plt.show()

