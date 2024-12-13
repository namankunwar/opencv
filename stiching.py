import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to detect keypoints and compute descriptors
def detect_and_compute(image, method="ORB"):
    if method == "ORB":
        detector = cv2.ORB_create()
    elif method == "SIFT":
        detector = cv2.SIFT_create()
    elif method == "SURF":
        detector = cv2.xfeatures2d.SURF_create()
    else:
        raise ValueError("Unknown method. Use 'ORB', 'SIFT', or 'SURF'")
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors

# Function to match features using BFMatcher
def match_features(desc1, desc2, method="ORB"):
    if method == "ORB":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:  # For SIFT or SURF
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

# Function to stitch images
def stitch_images(img1, img2, matches, kp1, kp2):
    # Extract matched keypoints
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Estimate Homography
    H, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Warp the second image to align with the first
    width = img1.shape[1] + img2.shape[1]
    height = img1.shape[0]
    panorama = cv2.warpPerspective(img2, H, (width, height))
    panorama[0:img1.shape[0], 0:img1.shape[1]] = img1
    return panorama

# Load and preprocess images
image1 = cv2.imread('photos/image_1.jpg')  
image2 = cv2.imread('photos/image_2.jpg')
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Step 1: Detect keypoints and compute descriptors
kp1, desc1 = detect_and_compute(gray1, method="ORB")
kp2, desc2 = detect_and_compute(gray2, method="ORB")

# Step 2: Match features
matches = match_features(desc1, desc2, method="ORB")

# Step 3: Stitch images with Homography and Blending
panorama = stitch_images(image1, image2, matches, kp1, kp2)

# Display results
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Panorama")
plt.show()
