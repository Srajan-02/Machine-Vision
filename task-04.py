import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the scanned old photograph
old_photo = cv2.imread("C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\old.jpeg")
plt.imshow(cv2.cvtColor(old_photo, cv2.COLOR_BGR2RGB))
plt.title('Scanned Old Photograph')
plt.show()


# Coordinates of key points on the distorted old photograph
distorted_points = np.array([[80, 120], [450, 120], [80, 350], [450, 350]])  


# Corresponding coordinates in the reference image
reference_points = np.array([[100, 100], [500, 100], [100, 400], [500, 400]]) 

# Compute the transformation matrix
transformation_matrix = cv2.getPerspectiveTransform(distorted_points.astype(np.float32), reference_points.astype(np.float32))

# Apply the transformation to the old photograph
rectified_photo = cv2.warpPerspective(old_photo, transformation_matrix, (600, 600))  

# Apply transformation with bilinear interpolation
rectified_photo_bilinear = cv2.warpPerspective(old_photo, transformation_matrix, (600, 600), flags=cv2.INTER_LINEAR)

# Display the original and rectified images
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(old_photo, cv2.COLOR_BGR2RGB))
plt.title('Original Scanned Photograph')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(rectified_photo_bilinear, cv2.COLOR_BGR2RGB))
plt.title('Rectified Photograph')
plt.show()
