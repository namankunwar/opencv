import cv2
import numpy as np

# Load the image

img = cv2.imread("photos/blank.jpg")
image = cv2.resize(img, (600,600), interpolation= cv2.INTER_AREA)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Simple Thresholding
_, threshold_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Adaptive Thresholding
adaptive_threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Display the results
cv2.imshow("Simple Threshold", threshold_image)
cv2.imshow("Adaptive Threshold", adaptive_threshold)

cv2.waitKey(0)
cv2.destroyAllWindows()
