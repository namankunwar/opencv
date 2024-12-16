import cv2 as cv
import numpy as np


# Load images in grayscale for processing
good_image = cv.imread('photos/good-image.jpg', cv.IMREAD_GRAYSCALE)
bad_image = cv.imread('photos/bad-image.jpg', cv.IMREAD_GRAYSCALE)

# Check if images are loaded properly
if good_image is None or bad_image is None:
    print("Error: Could not load images. Check file paths.")
    exit()

#Detect edges using Canny edge detector
# Canny edge detection is used for finding edges based on gradients.
# Arguments:
# - Image: Input grayscale image
# - Threshold1: Lower threshold for gradient strength
# - Threshold2: Upper threshold for gradient strength
# Apply binary threshold to convert the image into a binary image
#ret_1, threshold_1 = cv.threshold(good_image, 150, 255, cv.THRESH_BINARY)
#ret_2, threshold_2 = cv.threshold(bad_image, 150, 255, cv.THRESH_BINARY)

#Gaussian blur
#good_blur = cv.GaussianBlur(good_image, (5,5),0, cv.BORDER_DEFAULT)
#bad_blur = cv.GaussianBlur(bad_image, (5,5),0, cv.BORDER_DEFAULT)

#Bialateral blur
#good_blur= cv.bilateralFilter(good_image, 10, 25, 20)
#bad_blur= cv.bilateralFilter(bad_image, 10, 25, 20)
#median blur
good_blur= cv.medianBlur(good_image, 5)
#median blur
bad_blur= cv.medianBlur(bad_image, 5)

good_edges = cv.Canny(good_blur, 50, 150)
bad_edges = cv.Canny(bad_blur, 69, 160)

#Find contours of the edges
# Contours are the boundaries of the object detected.
# Arguments:
# - Image: Input binary image (e.g., edges from Canny)
# - Retrieval mode: cv.RETR_EXTERNAL to retrieve only the outermost contour
# - Approximation method: cv.CHAIN_APPROX_SIMPLE reduces the points in contours
contours_good, _ = cv.findContours(good_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
contours_bad, _ = cv.findContours(bad_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

#Draw the contours on the original images
# Create copies of the original images for contour visualization
good_image_color = cv.cvtColor(good_image, cv.COLOR_GRAY2BGR)  # Convert to color for drawing
bad_image_color = cv.cvtColor(bad_image, cv.COLOR_GRAY2BGR)

# Draw contours
# Arguments:
# - Image: Input image
# - Contours: List of contours detected
# - Contour index: -1 means draw all contours
# - Color: Color for drawing (B, G, R)
# - Thickness: Thickness of the lines
cv.drawContours(good_image_color, contours_good, -1, (0, 255, 0), 1)
cv.drawContours(bad_image_color, contours_bad, -1, (0, 255, 0), 1)

#Count the white pixels in the good image
# White pixels are defined as pixels with value 255 in a binary image
white_pixels_count = np.sum(good_image == 255)

# Display the results
print(f"Number of white pixels in the good image: {white_pixels_count}")

cv.imwrite('photos/contours_good_image.jpg', good_image_color)
cv.imwrite('photos/contours_bad_image.jpg', bad_image_color)

cv.imshow("Good Image", good_image_color)
cv.imshow("Bad Image", bad_image_color)
cv.waitKey(0) 
cv.destroyAllWindows()
