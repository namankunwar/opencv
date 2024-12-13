import cv2 as cv
import numpy as np

# Load a BGR image (color image)
image = cv.imread("photos/blank.jpg")
# Parameters:
# - "photos/sample.jpg": Path to the image file.
# - Returns an array representation of the image.

# Resize the image to 400x400 for uniformity
resized_img = cv.resize(image, (400, 400), interpolation=cv.INTER_AREA)
# Parameters:
# - image: Original image.
# - (400, 400): Desired dimensions (width, height).
# - interpolation=cv.INTER_AREA: Resizing technique suitable for shrinking the image.

# Create a circular mask
mask = np.zeros(resized_img.shape[:2], dtype="uint8")
# Creates a black image with the same height and width as the resized image but without color channels.

mask = cv.circle(mask, (200, 200), 100, 255, -1)
# Draws a white circle on the black mask.
# Parameters:
# - mask: The black mask image to draw on.
# - (200, 200): Center of the circle.
# - 100: Radius of the circle.
# - 255: Color of the circle (white in grayscale).
# - -1: Thickness, -1 means the circle will be filled.

# Apply the mask to the image using bitwise AND
masked_image = cv.bitwise_and(resized_img, resized_img, mask=mask)
# Parameters:
# - resized_img: Input image.
# - resized_img: Second input image (same as the first).
# - mask=mask: Applies the circular mask to the input image.
# Only the area of the image inside the white circle will be retained.

# Display the results
cv.imshow("Original Image", resized_img)  # Displays the resized original image
cv.imshow("Mask", mask)                   # Displays the circular mask
cv.imshow("Masked Image", masked_image)   # Displays the image with the mask applied

cv.waitKey(0)  # Waits indefinitely until a key is pressed
cv.destroyAllWindows()  # Closes all OpenCV windows
