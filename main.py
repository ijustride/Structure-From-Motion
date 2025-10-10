import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# this is a test project

# Focal lengths in pixels
f_x = 896
f_y = 576

# Optical center (principal point) in pixels
c_x = 1920
c_y = 960

# Camera intrinsic matrix
K = np.array([
    [f_x, 0, c_x],
    [0, f_y, c_y],
    [0, 0, 1]
])

# Initialize camera trajectory list
camera_positions = []

image_dir = 'pics/'
image_files = sorted(os.listdir(image_dir))

images = []

for image_file in image_files:
    # Construct full file path
    image_path = os.path.join(image_dir, image_file)
    
    # Load the image using OpenCV (grayscale or color, depending on your need)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Change to IMREAD_COLOR if needed
    
    # Append to the images list
    images.append(img)

# Load the list of images (add your image paths here)


# Initialize previous pose (for the first image, assume identity)
R_prev = np.eye(3)
T_prev = np.zeros((3, 1))

# Create a figure for 3D visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Loop through the images in pairs
for i in range(len(images) - 1):
    # Load the images in grayscale
    img1 = images[i]
    img2 = images[i + 1]

    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match the descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract points from keypoints based on the matches
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Draw the matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(int(0.03 * 1000))  # Display matches for 0.3 seconds

    # Find the Essential matrix
    E, mask = cv2.findEssentialMat(points1, points2, K)

    # Recover pose (rotation and translation)
    _, R, T, mask = cv2.recoverPose(E, points1, points2, K)

    # Update the current camera position
    # We apply the relative pose between the previous and current frame to the previous position
    T_current = T_prev + R_prev @ T  # New camera position
    camera_positions.append(T_current.flatten())

    # Update the previous pose
    R_prev = R
    T_prev = T_current

    # Plot the current camera position (trajectory)
    ax.scatter(T_current[0], T_current[1], T_current[2], c='r', marker='o')

# Format the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera Trajectory')

# Show the 3D trajectory plot
plt.show()
cv2.destroyAllWindows()
