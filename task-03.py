import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the distorted drone image
distorted_image = cv2.imread("C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\agriculture.jpeg")
plt.imshow(cv2.cvtColor(distorted_image, cv2.COLOR_BGR2RGB))
plt.title('Distorted Drone Image')
plt.show()

# Coordinates of control points in the distorted drone image
distorted_points = np.array([[150, 200], [600, 200], [150, 500], [600, 500]])  

# Corresponding coordinates in the rectified image
rectified_points = np.array([[100, 100], [500, 100], [100, 400], [500, 400]])  

# Compute the transformation matrix
transformation_matrix = cv2.getPerspectiveTransform(distorted_points.astype(np.float32), rectified_points.astype(np.float32))

# Apply the transformation to rectify the image
rectified_image = cv2.warpPerspective(distorted_image, transformation_matrix, (600, 600)) 

# Apply transformation using bilinear interpolation
rectified_image_bilinear = cv2.warpPerspective(distorted_image, transformation_matrix, (600, 600), flags=cv2.INTER_LINEAR)

# Display the distorted and rectified images side by side
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(distorted_image, cv2.COLOR_BGR2RGB))
plt.title('Distorted Drone Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(rectified_image_bilinear, cv2.COLOR_BGR2RGB))
plt.title('Rectified Image (Bilinear Interpolation)')
plt.show()
