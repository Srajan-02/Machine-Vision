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

# Load a color image
image_path = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\rgb.jpeg"
original_image = cv2.imread(image_path)

B, G, R = cv2.split(original_image)

# Applying transformations to each channel
negative_B = image_negative(B)
negative_G = image_negative(G)
negative_R = image_negative(R)

gamma_corrected_B = gamma_correction(B, 2.0)
gamma_corrected_G = gamma_correction(G, 2.0)
gamma_corrected_R = gamma_correction(R, 2.0)

log_transformed_B = log_transform(B)
log_transformed_G = log_transform(G)
log_transformed_R = log_transform(R)

# Merging the channels back together
negative_image = cv2.merge([negative_B, negative_G, negative_R])
gamma_corrected_image = cv2.merge([gamma_corrected_B, gamma_corrected_G, gamma_corrected_R])
log_transformed_image = cv2.merge([log_transformed_B, log_transformed_G, log_transformed_R])

# Display the original and the transformed images
plt.figure(figsize=(20, 10))

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Negative Image')
plt.imshow(cv2.cvtColor(negative_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Gamma Corrected (γ=2.0)')
plt.imshow(cv2.cvtColor(gamma_corrected_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Log Transformed Image')
plt.imshow(cv2.cvtColor(log_transformed_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
