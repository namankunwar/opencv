import cv2
import numpy as np

# Load the reference image (base image) and the image to align
# "cv2.IMREAD_COLOR" is the default flag, but other options include:
# cv2.IMREAD_GRAYSCALE (loads image in grayscale)
# cv2.IMREAD_UNCHANGED (loads image as is, including alpha channels)
img = cv2.imread('photos/reference_image.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('photos/target_image.jpg', cv2.IMREAD_COLOR)
ref_image = cv2.resize(img, (500,500), interpolation= cv2.INTER_AREA)
align_image = cv2.resize(img2, (500,500), interpolation= cv2.INTER_AREA)
# Convert both images to grayscale for feature detection
ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
align_gray = cv2.cvtColor(align_image, cv2.COLOR_BGR2GRAY)

# Initialize the ORB detector
# nfeatures: Maximum number of features to retain
# scoreType: Determines the ranking of keypoints (cv2.ORB_FAST_SCORE or cv2.ORB_HARRIS_SCORE)
orb = cv2.ORB_create(nfeatures=500, scoreType=cv2.ORB_FAST_SCORE)

# Step 1: Detect keypoints and compute descriptors using ORB (Oriented FAST and Rotated BRIEF)
# Keypoints: Points of interest in the image
# Descriptors: Vector representations of the keypoints for matching
keypoints1, descriptors1 = orb.detectAndCompute(ref_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(align_gray, None)

# Match features using the BFMatcher (Brute-Force Matcher)
# cv2.NORM_HAMMING is used for binary descriptors like ORB
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance (lower distance indicates better match)
matches = sorted(matches, key=lambda x: x.distance)

# Draw the top matches for visualization
matched_image = cv2.drawMatches(ref_image, keypoints1, align_image, keypoints2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('Matches', matched_image)

# Extract location of good matches
points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

# Estimate homography matrix using RANSAC
# RANSAC: Random Sample Consensus, robust to outliers
H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

# Warp the align_image to the reference image's perspective
aligned_image = cv2.warpPerspective(align_image, H, (ref_image.shape[1], ref_image.shape[0]))

# Display the results
cv2.imshow('Reference Image', ref_image)
cv2.imshow('Aligned Image', aligned_image)

# Save the aligned image to disk
cv2.imwrite('photos/aligned_output.jpg', aligned_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Explanation of Parameters:
# - cv2.ORB_create(nfeatures=500): nfeatures sets the maximum number of features to detect.
# - cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True): NORM_HAMMING is optimal for ORB's binary descriptors.
# - cv2.findHomography(points2, points1, cv2.RANSAC, 5.0):
#   - Method: cv2.RANSAC ensures robust estimation by ignoring outliers.
#   - RANSAC Reprojection Threshold: 5.0 (adjust based on alignment accuracy).
# - cv2.warpPerspective(): Applies the homography to warp the image.

# Other Options:
# - For feature detection, alternatives include SIFT (cv2.SIFT_create) and SURF (cv2.xfeatures2d.SURF_create).
# - For matching, FLANN-based matcher (cv2.FlannBasedMatcher) can be used for faster performance on large datasets.
#
# Why Use Feature Detection and Alignment:
# - Aligning images is crucial in tasks like stitching panoramas, object tracking, and template matching.
# - ORB is efficient for real-time applications, balancing speed and accuracy.
