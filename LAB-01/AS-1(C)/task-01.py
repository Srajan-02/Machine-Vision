import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image in grayscale
image = cv2.imread("C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\gray.jpeg", cv2.IMREAD_GRAYSCALE)

# Display the original image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Plot histogram of the original image
plt.subplot(1, 2, 2)
plt.hist(image.ravel(), bins=256, range=[0, 256], color='black')
plt.title('Histogram of Original Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# Calculate histogram manually
histogram = np.zeros(256, dtype=int)

for pixel in image.ravel():
    histogram[pixel] += 1

# Plot the calculated histogram
plt.figure(figsize=(6, 4))
plt.bar(range(256), histogram, color='black', width=1.0)
plt.title('Calculated Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()


# Compute the CDF
cdf = histogram.cumsum()

# Normalize the CDF to the range [0, 255]
cdf_normalized = ((cdf - cdf.min()) / (cdf.max() - cdf.min())) * 255
cdf_normalized = cdf_normalized.astype(np.uint8)

# Plot the CDF
plt.figure(figsize=(6, 4))
plt.plot(cdf_normalized, color='blue')
plt.title('Cumulative Distribution Function (CDF)')
plt.xlabel('Pixel Intensity')
plt.ylabel('Cumulative Frequency')
plt.grid()
plt.show()


# Apply histogram equalization
equalized_image = cdf_normalized[image]

# Display the equalized image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

# Plot histogram of the equalized image
plt.subplot(1, 2, 2)
plt.hist(equalized_image.ravel(), bins=256, range=[0, 256], color='black')
plt.title('Histogram of Equalized Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()



