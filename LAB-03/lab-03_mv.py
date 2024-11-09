import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load Video
video_path = "C:\\Users\\sraja\\Downloads\\7565438-hd_1080_1920_25fps.mp4"
cap = cv2.VideoCapture(video_path)
frame_list = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_list.append(frame)

cap.release()
lower_threshold = np.array([100, 150, 0])   
upper_threshold = np.array([140, 255, 255]) 

# Spatio-Temporal Segmentation
segmented_frames = []
for frame in frame_list:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_threshold, upper_threshold) 
    segmented_frames.append(mask)


hard_cut_threshold = 500000
scene_cuts = []
for i in range(1, len(frame_list)):
    diff = cv2.absdiff(frame_list[i], frame_list[i-1])
    if np.sum(diff) > hard_cut_threshold:
        scene_cuts.append(i)

for idx in scene_cuts:
    plt.imshow(cv2.cvtColor(frame_list[idx], cv2.COLOR_BGR2RGB))
    plt.title(f'Scene Cut Detected at Frame {idx}')
    plt.show()
