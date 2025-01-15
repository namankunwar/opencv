import numpy as np
import cv2

def draw_circles_in_roi(image, roi, step_size):

    try:
        # Extract ROI dimensions
        x, y, width, height = roi

        # Validate inputs
        if step_size <= 0:
            raise ValueError("Step size must be a positive integer.")
        if width <= 0 or height <= 0:
            raise ValueError("ROI dimensions must be positive.")
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a valid numpy array.")

        # Initialize output image
        output_image = image.copy()

        # List to store circle centers
        circle_centers = []

        # Start looping through the ROI in step_size increments
        current_x = x
        while current_x <= x + width:
            current_y = y
            while current_y <= y + height:
                # Add circle center to the list
                circle_centers.append((int(current_x), int(current_y)))

                # Draw the circle
                cv2.circle(output_image, (int(current_x), int(current_y)), step_size // 2, (0, 255, 0), 2)

                # Move to the next step in y-direction
                current_y += step_size

            # Move to the next step in x-direction
            current_x += step_size

        return output_image, circle_centers

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return image, []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return image, []


image = np.zeros((500, 500, 3), dtype=np.uint8)
roi = (50, 50, 300, 200)  # x, y, width, height
step_size = 50

output_image, centers = draw_circles_in_roi(image, roi, step_size)

# Display the result
cv2.imshow("Circles in ROI", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Circle Centers:", centers)
