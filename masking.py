import cv2 as cv
import numpy as np

# Create two blank (black) images of size 300x300 with single channel (grayscale)
blank1 = np.zeros((300, 300), dtype="uint8")  # First blank image
blank2 = np.zeros((300, 300), dtype="uint8")  # Second blank image

# Draw a white rectangle on the first blank image
rectangle = cv.rectangle(blank1.copy(), (50, 50), (250, 250), 255, -1)
# Parameters:
# - blank1.copy(): Copy of the blank image to draw the rectangle on.
# - (50, 50): Top-left corner of the rectangle.
# - (250, 250): Bottom-right corner of the rectangle.
# - 255: Color of the rectangle (white in grayscale).
# - -1: Thickness, -1 means the rectangle will be filled.

# Draw a white circle on the second blank image
circle = cv.circle(blank2.copy(), (150, 150), 100, 255, -1)
# Parameters:
# - blank2.copy(): Copy of the blank image to draw the circle on.
# - (150, 150): Center of the circle.
# - 100: Radius of the circle.
# - 255: Color of the circle (white in grayscale).
# - -1: Thickness, -1 means the circle will be filled.

# Perform bitwise AND operation
bitwise_and = cv.bitwise_and(rectangle, circle)
# Keeps only the overlapping region of the rectangle and circle.

# Perform bitwise OR operation
bitwise_or = cv.bitwise_or(rectangle, circle)
# Combines all white regions from both the rectangle and circle.

# Perform bitwise XOR operation
bitwise_xor = cv.bitwise_xor(rectangle, circle)
# Highlights only the non-overlapping regions of the rectangle and circle.

# Perform bitwise NOT operation on the rectangle
bitwise_not_rect = cv.bitwise_not(rectangle)
# Inverts the colors of the rectangle (black becomes white and vice versa).

# Display the original shapes and the results of the bitwise operations
cv.imshow("Rectangle", rectangle)          # Displays the rectangle
cv.imshow("Circle", circle)                # Displays the circle
cv.imshow("Bitwise AND", bitwise_and)      # Displays the result of AND
cv.imshow("Bitwise OR", bitwise_or)        # Displays the result of OR
cv.imshow("Bitwise XOR", bitwise_xor)      # Displays the result of XOR
cv.imshow("Bitwise NOT (Rectangle)", bitwise_not_rect)  # Displays the result of NOT

cv.waitKey(0)  # Waits indefinitely until a key is pressed
cv.destroyAllWindows()  # Closes all OpenCV windows
