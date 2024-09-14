import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the distorted satellite image
distorted_image = cv2.imread("C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\satellite.jpeg")
plt.imshow(cv2.cvtColor(distorted_image, cv2.COLOR_BGR2RGB))
plt.title('Distorted Satellite Image')
plt.show()

# Coordinates of GCPs in distorted image
distorted_points = np.array([[100, 150], [400, 150], [100, 400], [400, 400]])

# Coordinates of corresponding GCPs in rectified image
rectified_points = np.array([[0, 0], [300, 0], [0, 300], [300, 300]])


# Compute the transformation matrix
transformation_matrix = cv2.getPerspectiveTransform(distorted_points.astype(np.float32), rectified_points.astype(np.float32))

# Apply the perspective transformation
rectified_image = cv2.warpPerspective(distorted_image, transformation_matrix, (300, 300))


# Apply transformation with bilinear interpolation
rectified_image_bilinear = cv2.warpPerspective(distorted_image, transformation_matrix, (300, 300), flags=cv2.INTER_LINEAR)

# Show the rectified image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(distorted_image, cv2.COLOR_BGR2RGB))
plt.title('Distorted Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(rectified_image_bilinear, cv2.COLOR_BGR2RGB))
plt.title('Rectified Image with Bilinear Interpolation')
plt.show()

