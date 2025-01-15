import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import time
import math
from matplotlib.cm import tab10
import random
import warnings
from scipy.signal import savgol_filter
from numpy.linalg import norm
from math import acos, degrees
from file import Measurement

class TurningPoints:
    """
    A class to encapsulate various contour processing methods such as:
    - Reading and thresholding images
    - Finding contours and smoothing them with Savitzky-Golay
    - Traversing in ROI
    - Computing angles between vectors
    - Finding turning points
    - Demonstrating the overall pipeline
    """
    
    def __init__(self, image_path):
        """
        Initialize the processor with a given image path.
        The grayscale image is stored as self.image.
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise ValueError(f"Could not read the image from {image_path}")

    @staticmethod
    def savitzky_golay(y, window_size, poly_order, deriv=0, rate=1):
        """
        Apply a Savitzky-Golay filter to smooth or compute the derivative of a 1D signal.
        """
        if window_size < 1:
            raise ValueError("window_size must be >= 1.")
        if window_size % 2 == 0:
            window_size += 1
            warnings.warn("window_size must be odd; incrementing by 1.", UserWarning)
        if poly_order >= window_size:
            raise ValueError("poly_order must be less than window_size.")

        half_window = (window_size - 1) // 2

        # Build a matrix of powers
        order_range = range(poly_order + 1)
        b = np.array([[k ** i for i in order_range] 
                      for k in range(-half_window, half_window + 1)])
        # Compute the pseudo-inverse and select the row for the desired derivative
        m = np.linalg.pinv(b)[deriv] * rate ** deriv * factorial(deriv)

        # Pad the signal
        firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
        lastvals =  y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])
        y_padded = np.concatenate((firstvals, y, lastvals))

        # Convolve with filter coefficients
        return np.convolve(m[::-1], y_padded, mode='valid')

    @staticmethod
    def smooth_and_filter_xy(
        x, 
        y, 
        threshold=2.0,
        sg_window_1=31, 
        sg_window_2=21, 
        poly_order=2
    ):
        """
        Smooth a pair of 1D signals (x and y) using Savitzky-Golay filtering and
        remove minor jitter using a threshold-based filter on consecutive differences.
        Ensures the final contour is closed (no gap between first and last points).
        """
        x = np.asarray(x)
        y = np.asarray(y)

        # -- Ensure the contour is closed by appending first point if needed
        if (x[0] != x[-1]) or (y[0] != y[-1]):
            x = np.append(x, x[0])
            y = np.append(y, y[0])

        half_window = (sg_window_1 - 1) // 2
        # Wrap-around padding
        x_padded = np.concatenate([x[-half_window:], x, x[:half_window]])
        y_padded = np.concatenate([y[-half_window:], y, y[:half_window]])

        # Smooth the padded signals via Savitzky-Golay
        x_smooth_padded = TurningPoints.savitzky_golay(x_padded, sg_window_1, poly_order)
        y_smooth_padded = TurningPoints.savitzky_golay(y_padded, sg_window_1, poly_order)

        # Remove the padding
        smoothed_x = x_smooth_padded[half_window : -half_window]
        smoothed_y = y_smooth_padded[half_window : -half_window]    

        # Compute absolute differences
        diff_x = np.abs(np.diff(smoothed_x))
        diff_y = np.abs(np.diff(smoothed_y))

        filtered_x = np.copy(smoothed_x)
        filtered_y = np.copy(smoothed_y)

        # Simple threshold-based filtering
        for i in range(1, len(smoothed_x) - 1):
            if diff_x[i - 1] < threshold and diff_x[i] < threshold:
                filtered_x[i] = 0.5 * (smoothed_x[i - 1] + smoothed_x[i + 1])
            if diff_y[i - 1] < threshold and diff_y[i] < threshold:
                filtered_y[i] = 0.5 * (smoothed_y[i - 1] + smoothed_y[i + 1])

        # Final smoothing pass
        filtered_x = TurningPoints.savitzky_golay(filtered_x, sg_window_2, poly_order)
        filtered_y = TurningPoints.savitzky_golay(filtered_y, sg_window_2, poly_order)

        # Re-close the contour explicitly
        filtered_x[-1] = filtered_x[0]
        filtered_y[-1] = filtered_y[0]

        return filtered_x, filtered_y
    @staticmethod    
    def reorder_contour_by_roi(contour, roi):
        """
        Reorder a contour so it starts at the point closest to the center of the ROI.

        Args:
        - contour: np.ndarray, the contour to reorder, should be a list of points.
        - roi: tuple, (x, y, width, height) defining the Region of Interest.

        Returns:
        - reordered_contour: np.ndarray, the contour starting at the closest point to the ROI center.
        """
        x, y, w, h = roi
        roi_center = (x + w // 2, y + h // 2)  # Center of the ROI
        
        # Find the point on the contour closest to the ROI center
        distances = [cv2.norm(pt[0] - np.array(roi_center)) for pt in contour]
        new_start_idx = np.argmin(distances)

        # Reorder the contour to start at the closest point
        reordered_contour = np.concatenate((contour[new_start_idx:], contour[:new_start_idx]))
        return reordered_contour

    @staticmethod
    def transform_points(x1, y1, x2, y2):
        """
        Takes 2 points as input and aligns them in a straight line by taking average.
        """
        avg_x = (x1 + x2) / 2
        return (avg_x, y1), (avg_x, y2)

    def get_contour_coordinates(self):
        """
        Reads the image from self.image, finds the largest contour,
        smooths and filters it, and returns its coordinates.
        """
        if self.image is None:
            raise ValueError("Error: self.image is None. Please check the image path.")

        _, binary = cv2.threshold(self.image, 250, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("Error: No contours found.")
            return None

        contour = max(contours, key=cv2.contourArea)[:, 0, :]  # largest contour

        roi = (450, 145, 50, 50)  # A square ROI inside the rectangle
        # Reorder the contour to start from the midpoint of the ROI
        contour = TurningPoints.reorder_contour_by_roi(contour, roi)
        x, y = contour[:, 0], contour[:, 1]

        # Optionally define a desired point to reorder or compute start_index
        # For demonstration, we skip reordering here.

        # Smooth and filter
        x, y = TurningPoints.smooth_and_filter_xy(x, y)
        contour = np.column_stack((x, y))
        return contour

    


    @staticmethod
    def traverse_contour_in_roi(contour, roi):
        """
        Traverse the given contour once and return the points that lie
        within the specified ROI in the original order.
        """
        if contour is None or len(contour) == 0:
            return []

        (rx, ry, rw, rh) = roi
        traversed_points = []

        for px, py in zip(contour[:, 0], contour[:, 1]):
            if (rx <= px < rx + rw) and (ry <= py < ry + rh):
                traversed_points.append((px, py))

        return traversed_points

    @staticmethod
    def calculate_angle_between_vectors(point1, intersection, point2):
        """
        Calculate the angle between two vectors formed by three points.
        """
        vector1 = (point1[0] - intersection[0], point1[1] - intersection[1])
        vector2 = (point2[0] - intersection[0], point2[1] - intersection[1])

        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

        if magnitude1 == 0 or magnitude2 == 0:
            raise ValueError("One of the vectors has zero length.")

        cos_theta = dot_product / (magnitude1 * magnitude2)
        # Clamp for numerical stability
        cos_theta = max(-1, min(1, cos_theta))
        angle_radians = math.acos(cos_theta)
        angle_degrees = math.degrees(angle_radians)

        # Example heuristic
        if angle_degrees <= 45:
            return angle_degrees + 180
        else:
            return angle_degrees

    @staticmethod
    def make_data_vectorable(a, b):
        """
        Combines two 1D arrays into a list of triplets of consecutive points.
        """
        a = np.squeeze(np.array(a))
        b = np.squeeze(np.array(b))
        if len(a) != len(b):
            raise ValueError("Both arrays must have the same length.")

        result = np.column_stack((a, b))
        arr = np.array(result, dtype=float)

        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("Input array must be of shape (n, 2).")
        n = arr.shape[0]
        if n < 3:
            raise ValueError("Input array must have at least 3 rows.")

        return [
            (
                tuple(arr[i]),
                tuple(arr[i + 1]),
                tuple(arr[i + 2])
            )
            for i in range(n - 2)
        ]
    
    @staticmethod
    def get_points_at_distance(points, start_index=0, distance=20):
        """
        Traverse through a list of points and return points at a specified distance.
        
        Parameters:
        - points: List of (x, y) tuples or numpy array of shape (n, 2) representing the contour.
        - start_index: Index to start the traversal from.
        - distance: The distance between successive points to be extracted.
        
        Returns:
        - result_points: List of points meeting the distance criteria.
        """
        points = np.array(points)
        if len(points) <= 1:
            return points.tolist()  # Return input if there's only one or no point
        
        result_points = [tuple(points[start_index])]
        current_point = points[start_index]
        total_points = len(points)
        
        for i in range(start_index + 1, total_points):
            next_point = points[i]
            dist = np.linalg.norm(next_point - current_point)
            
            if dist >= distance:
                result_points.append(tuple(next_point))
                current_point = next_point
        
        # Ensure the last point is included
        if tuple(points[-1]) not in result_points:
            result_points.append(tuple(points[-1]))
        
        return result_points

    @staticmethod
    def compute_turning_points(points, step_size):
        """
        Computes coarser derivatives by averaging over specified intervals.
        Returns a dict {(pt1, pt2, pt3): angle, ...}.
        """
        start = time.time()

        # Sort by x for demonstration (optional)
        points_sorted =points#sorted(points, key=lambda point: point[0])
        x, y = zip(*points_sorted)
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        def get_points(x,y,step_size):
            n = len(x)
            if n < 2:
                raise ValueError("At least two points are required.")
            if not 0 < step_size <= 1:
                step=step_size
            else:    
                # Step size in terms of number of points
                step = max(1, int(step_size * n))
                step = min(step, n - 1)

            indices = np.arange(0, n, step)
            x_new = x[indices]
            y_new = y[indices]
            return x_new,y_new
        def rearrange(x,x1,x2):
            index1 = np.where(x == x1)[0][0]
            index2 = np.where(x == x2)[0][0]
            # Ensure the indices are in the correct order
            start, end = min(index1, index2), max(index1, index2)

            # Slice the array
            result = x[start:end + 1]
            # print(result)
            return result 

        x_new,y_new=get_points(x,y,0.3)    
        x1,x2=x_new[0],x_new[-1]
        x=rearrange(x,x1,x2)
    
        y1,y2=y_new[0],y_new[-1]
        y=rearrange(y,y1,y2)
        # print(x)
        point=np.column_stack((x,y))
        point_new=TurningPoints.get_points_at_distance(point,0,40)  
        image=cv2.imread('fick.jpg')
        measure=Measurement(image)
        
        for point in point_new:
            measure.add_point(point)
            # measure.draw()
        # print(x_new)
        

        vectors = TurningPoints.make_data_vectorable(x_new, y_new)
        vector_angle_dict = {}
        for vector in vectors:
            angle = TurningPoints.calculate_angle_between_vectors(*vector)
            vector_angle_dict[vector] = angle

        print("Computation time (seconds):", time.time() - start)
        return vector_angle_dict

    def flow(self):
        start=time.time()
        """
        Demonstration of the class usage:
            1) Get the main contour from the provided image.
            2) Define an ROI and traverse points within that ROI.
            3) Compute turning points in the ROI.
            4) Visualize results.
        """
        # 1. Get contour from self.image
        contour = self.get_contour_coordinates()
        if contour is None:
            return

        print("Contour shape:", contour.shape)

        # 2. Define an ROI (x, y, w, h)
        roi = (850, 100, 100, 200)  
        points_in_roi = TurningPoints.traverse_contour_in_roi(contour, roi)
        print(f"Number of points in ROI: {len(points_in_roi)}")

        # 3. Draw for visualization
        color_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        for (cx, cy) in contour:
            cv2.circle(color_image, (int(cx), int(cy)), 2, (0, 255, 0), -1)
        rx, ry, rw, rh = roi
        cv2.rectangle(color_image, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 1)
        cv2.imshow("Contour with ROI", color_image)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 4. Compute turning points in the ROI
        step_size = 0.30
        derivative = TurningPoints.compute_turning_points(points_in_roi, step_size)
        if not derivative:
            print("No turning points calculated (possibly too few points in ROI).")
            return

        # 5. Find a minimal angle example (just demonstration)
        key_with_min_value = min(derivative, key=derivative.get)
        print(f"Min angle is {derivative[key_with_min_value]} at points {key_with_min_value}")

        # If you have a Measurement class and want to plot them
        measure=Measurement(self.image)
        for point in key_with_min_value:
            measure.add_point(point)
        # measure.draw()

        for points_triplet in derivative.keys():
            for pt in points_triplet:
                measure.add_point(pt)
            # measure.draw()

        print(print("Computation time (seconds):", time.time() - start))

# -----------------
# Example usage:
if __name__ == "__main__":
    processor = TurningPoints("fick.jpg")
    processor.flow()
