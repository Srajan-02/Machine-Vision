import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "C:\\Users\\sraja\\OneDrive\\Documents\\Srajan\\Machine_Vision\\Images\\rgb.jpeg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
else:
    def compute_statistics(image, channel_name):
        mean = np.mean(image)
        std_dev = np.std(image)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        return mean, std_dev, hist
    
    channels = cv2.split(image)
    channel_names = ['Blue', 'Green', 'Red']
    
    for i, channel in enumerate(channels):
        mean, std_dev, hist = compute_statistics(channel, channel_names[i])
        print(f"{channel_names[i]} channel - Mean: {mean:.2f}, Std Dev: {std_dev:.2f}")
        
        plt.figure()
        plt.title(f'{channel_names[i]} Histogram')
        plt.xlabel('Intensity Value')
        plt.ylabel('Count')
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.show()
    
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    
    def display_image(image, title):
        plt.figure()
        plt.title(title)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    
    display_image(image, 'Original Image')
    display_image(image_hsv, 'HSV Color Space')
    display_image(image_lab, 'Lab Color Space')
