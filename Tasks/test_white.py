import cv2
import numpy as np

# Load the apple image
apple_img = cv2.imread("apple_image.jpg")  # Make sure the image path is correct
apple_img = cv2.resize(apple_img, (512, 512))  # Resize apple image to 512x512

# Create a blank image with the size of 4 rows and 7 columns of 512x512 images
blank_image = np.zeros((4 * 512, 7 * 512, 3), dtype=np.uint8)

# Define the sequence of the apple image positions in the 4x7 grid (array1)
array1 = [25, 26, 27, 28, 21, 14, 7, 6, 5, 4, 3, 2, 1, 8, 15, 22, 23, 24]

# Grid size
rows, cols = 4, 7
pixel_size = 512  # Size of each apple image (512x512)

# Place the apple images in sequential order in the blank image
for idx in array1:
    # Convert the linear index to row and column in a 4x7 grid
    row = (idx - 1) // cols  # Calculate the row (index starts from 1, so subtract 1)
    col = (idx - 1) % cols   # Calculate the column
    
    # Calculate the position to place the apple image
    x_start = col * pixel_size
    y_start = row * pixel_size
    
    # Place the apple image at the calculated position
    blank_image[y_start:y_start + pixel_size, x_start:x_start + pixel_size] = apple_img
    
    # Resize the blank image to fit within a smaller window while maintaining aspect ratio
    height, width = blank_image.shape[:2]
    max_dim = 700  # Maximum dimension for the resized image (width or height)
    
    # Calculate the scale factor to fit within max_dim
    scale_factor = min(max_dim / width, max_dim / height)
    
    # Resize the image
    resized_image = cv2.resize(blank_image, (int(width * scale_factor), int(height * scale_factor)))

    # Show the resized image
    cv2.imshow("Animation", resized_image)
    
    # Introduce a small delay for the animation effect
    cv2.waitKey(200)  # Adjust the delay as per your preference

# Close the window after animation is complete
cv2.destroyAllWindows()
