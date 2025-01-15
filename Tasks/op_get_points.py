import numpy as np
import cv2

class TurningPoints:
    def __init__(self, img_path):
        """Reads and stores the image after converting to grayscale."""
        self.image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise ValueError("Image not found at provided path.")
        self.contours = self._find_contours(self.image)

    def _find_contours(self, img):
        """Finds contours from a binary image and excludes the outer border."""
        # Convert image to binary (thresholding)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Find all contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)



    def get_contour_coordinates(self):
        """Returns the x, y coordinates of the contour."""
        contour = self.contours[0].squeeze()  # Take the first valid contour

        # Ensure the contour is 2D (if it's a 1D array, reshape it)
        if contour.ndim == 1:
            contour = contour.reshape(-1, 2)

        return contour[:, 0], contour[:, 1]

    def get_contour_distance(self, contour):
        """Calculate cumulative distance along the contour."""
        dist = [0]
        for i in range(1, len(contour)):
            dist.append(dist[i-1] + np.linalg.norm(contour[i] - contour[i-1]))
        return np.array(dist)

    def get_points_within_roi(self, x, y, roi_x_min, roi_x_max, roi_y_min, roi_y_max):
        """Returns points within a specified ROI from the contour."""
        points_in_roi = []
        for xi, yi in zip(x, y):
            if roi_x_min <= xi <= roi_x_max and roi_y_min <= yi <= roi_y_max:
                points_in_roi.append((xi, yi))
        return points_in_roi

    def generate_step_points(self, roi_x_min, roi_x_max, roi_y_min, roi_y_max, num_steps=10):
        """Generate equidistant points along the contour inside the ROI."""
        # Get contour coordinates
        x, y = self.get_contour_coordinates()

        # Filter points inside the ROI
        points_in_roi = self.get_points_within_roi(x, y, roi_x_min, roi_x_max, roi_y_min, roi_y_max)
        
        if not points_in_roi:
            print("Error: No points inside the ROI.")
            return []

        # Calculate the total distance along the contour
        contour = np.array(points_in_roi)
        distances = self.get_contour_distance(contour)

        # Total contour length and step size
        total_length = distances[-1]
        step_size = total_length / (num_steps - 1)

        # Generate points at equidistant intervals
        step_points = []
        for i in range(1, num_steps):
            target_distance = i * step_size
            closest_point_index = np.argmin(np.abs(distances - target_distance))
            step_points.append(contour[closest_point_index])

        return np.array(step_points)

    def visualize_contour_with_steps(self, roi_x_min, roi_x_max, roi_y_min, roi_y_max, num_steps=10):
        """Visualize the contour and equidistant points in the ROI."""
        image_copy = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

        # Draw the contours (excluding the border)
        for contour in self.contours:
            # Ensure we are not drawing the border contours by checking their size
            if cv2.contourArea(contour) > 100:  # Threshold area to exclude small contours
                cv2.drawContours(image_copy, [contour], -1, (0, 255, 0), 2)

        # Draw ROI rectangle
        cv2.rectangle(image_copy, (roi_x_min, roi_y_min), (roi_x_max, roi_y_max), (255, 0, 0), 2)

        # Generate equidistant points inside the ROI
        step_points = self.generate_step_points(roi_x_min, roi_x_max, roi_y_min, roi_y_max, num_steps)
        
        # Draw circles around each step point
        for (xi, yi) in step_points:
            cv2.circle(image_copy, (int(xi), int(yi)), 5, (0, 0, 255), -1)

        # Show the result
        cv2.imshow("Contour with Step Points", image_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Usage Example
img_path = 'fick.jpg'
turning_points = TurningPoints(img_path)

# Define the ROI coordinates
roi_x_min, roi_x_max, roi_y_min, roi_y_max = 50, 200, 50, 200

# Visualize the contour with equidistant points in the ROI
turning_points.visualize_contour_with_steps(roi_x_min, roi_x_max, roi_y_min, roi_y_max, num_steps=10)
