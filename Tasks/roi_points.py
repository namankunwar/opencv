
def roi_points(roi, steps):
    
    try:
        # Extract ROI dimensions
        x, y, width, height = roi

        # Validate inputs
        if steps <= 0:
            raise ValueError("Number of steps must be a positive integer.")
        if width <= 0 or height <= 0:
            raise ValueError("ROI dimensions must be positive.")

        start = (x, y)
        end_x = x + width
        end_y = y + height

        # Step increments
        step_x = width / steps
        step_y = height / steps

        points = []

        # Generate points starting from (x, y)
        current_x = x
        current_y = y

        for i in range(steps + 1):  # Include the last step
            # Add the current point to the array
            points.append((current_x, current_y))

            # Update coordinates
            current_x += step_x
            current_y += step_y

            # Handle errors: Ensure the point is inside the ROI
            if current_x > end_x or current_y > end_y:
                break  # Prevent going out of bounds

        return points

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


roi = (10, 20, 100, 50)  # x, y, width, height
steps = 10
points = roi_points(roi, steps)
print("Computed Points:", points)
