import cv2
import numpy as np

# Load the image
img = cv2.imread("photos/blank.jpg")
image= cv2.resize(img, (500,500),interpolation=cv2.INTER_AREA)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Laplacian Edge Detection
laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

# Sobel Edge Detection
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # x-direction
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # y-direction
sobel_magnitude = cv2.bitwise_or(sobel_x, sobel_y)  # Combining x and y gradients

# Display the results
cv2.imshow("Laplacian Edge Detection", laplacian)
cv2.imshow("Sobel Edge Magnitude", sobel_magnitude)

cv2.waitKey(0)
cv2.destroyAllWindows()
