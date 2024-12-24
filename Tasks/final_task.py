import cv2 as cv
import numpy as np
import os

# Function to calculate the number of white pixels in an image (after edge detection)
def count_white_pixels(image):
    # Count the number of white pixels (edges)
    white_pixels = np.sum(image == 255)
    return white_pixels

# Function to classify image based on the number of white pixels (edges)
def classify_image_by_white_pixels(white_pixel_count, threshold=1345):
    if white_pixel_count < threshold:
        return "Bad - (low white pixels)"
    else:
        return "Good Image"

# Path to the folder containing images
image_folder = 'Tasks/logo/'

# Loop through all images in the folder
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the image
        image_path = os.path.join(image_folder, filename)
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Error: Could not load image {filename}. Skipping...")
            continue
        
        # Median blur to reduce noise
        blurred_image = cv.medianBlur(image, 15)

        # Canny edge detection
        edges = cv.Canny(blurred_image, 70, 150)

        # Count white pixels (edges) in the Canny edge-detected image
        white_pixel_count = count_white_pixels(edges)

        # Classify the image based on the number of white pixels
        classification = classify_image_by_white_pixels(white_pixel_count)

        print(f"Image: {filename} - {classification} (White Pixels: {white_pixel_count})")

        # Convert the image to color for drawing contours
        image_color = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

        # Draw contours on the image
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(image_color, contours, -1, (0, 255, 0), 1)

        # Save the image with contours drawn
        #output_image_path = os.path.join('Tasks/logo', f'contours_{filename}')
        #cv.imwrite(output_image_path, image_color)

        # Optionally display the image
        cv.imshow(f"Processed Image: {filename}", image_color)

cv.waitKey(0)
cv.destroyAllWindows()
