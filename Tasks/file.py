import math
from numpy.linalg import norm
from math import acos, degrees
import numpy as np
import random
import cv2

class Measurement:
    def __init__(self, image=None):
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid image provided.")
        self.lines = {}
        self.line_equations = {}
        self.points = []
        self.radius = 5
        self.thickness = 2
        self.image = image
        self.line_colors = {}  # Cache colors for lines

    def generate_random_color(self):
        return tuple(random.randint(0, 255) for _ in range(3))

    def add_point(self, point):
        if point not in self.points:
            self.points.append(point)
            


    def draw(self):
        """
        Plots points, lines, and optionally draws lines from slope-intercept equations
        with shared colors for connected elements.
        """
        # Ensure image is available
        if self.image is None:
            raise ValueError("No image provided for drawing.")

        # Make a copy of the image to avoid altering the original
        output_image = self.image.copy()

        # Draw lines and connected points
        if self.lines:
            for (start_point, end_point), distance in self.lines.items():
                if (start_point, end_point) not in self.line_colors:
                    self.line_colors[(start_point, end_point)] = self.generate_random_color()
                color = self.line_colors[(start_point, end_point)]

                # Draw line
                cv2.line(output_image, tuple(map(int, start_point)), tuple(map(int, end_point)), color, self.thickness)

                # Calculate midpoint
                midpoint = (
                    int((start_point[0] + end_point[0]) / 2),
                    int((start_point[1] + end_point[1]) / 2)
                )

                # Display distance
                if distance is not None:
                    text_size = cv2.getTextSize(f"{distance:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    text_x, text_y = midpoint[0], midpoint[1]
                    cv2.rectangle(
                        output_image,
                        (text_x, text_y - text_size[1] - 5),
                        (text_x + text_size[0] + 5, text_y + 5),
                        (0, 0, 0),  # Background color
                        -1
                    )
                    cv2.putText(
                        output_image,
                        f"{distance:.2f}",
                        (text_x + 2, text_y - 2),  # Slightly offset text within the rectangle
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA
                    )


                # Draw points
                cv2.circle(output_image, tuple(map(int, start_point)), self.radius, color, -1)
                cv2.circle(output_image, tuple(map(int, end_point)), self.radius, color, -1)


        # Draw isolated points
        for point in self.points:
            cv2.circle(output_image, tuple(map(int, point)), self.radius, self.generate_random_color(), -1)

        # Draw line equations
        for (line1, line2), (intersection_data, angle) in self.line_equations.items():
            if line1 is None:  # Skip empty or invalid entries
                continue

            slope1, intercept1 = line1
            
            color1 = self.generate_random_color()

            # Draw the first line
            def draw_line(slope, intercept, color):
                if math.isinf(slope):  # Vertical line
                    x = int(intercept)
                    cv2.line(output_image, (x, 0), (x, output_image.shape[0] - 1), color, self.thickness)
                else:
                    x_start, x_end = 0, output_image.shape[1] - 1
                    y_start = int(slope * x_start + intercept)
                    y_end = int(slope * x_end + intercept)
                    cv2.line(output_image, (x_start, y_start), (x_end, y_end), color, self.thickness)

            draw_line(slope1, intercept1, color1)

            if line2 is not None:  # Handle the second line if it exists
                slope2, intercept2 = line2
                color2 = self.generate_random_color()
                draw_line(slope2, intercept2, color2)

                # Draw intersection point if valid
                if intersection_data and not any(map(math.isnan, intersection_data)):
                    x_intersect, y_intersect = map(int, intersection_data)
                    if 0 <= x_intersect < output_image.shape[1] and 0 <= y_intersect < output_image.shape[0]:
                        # Draw the intersection point
                        cv2.circle(output_image, (x_intersect, y_intersect), self.radius, (0, 255, 0), -1)

                        # Display angle at the intersection point
                        angle_text = f"{angle:.2f}deg"   
                        x_offset=int(x_intersect+0.09*self.image.shape[0] )    
                        y_offset=int(y_intersect+0.01*self.image.shape[1])    
                        print(x_offset)
                        text_position = (x_offset, y_offset)  # Offset the text for better visibility
                        text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        cv2.rectangle(
                            output_image,
                            (text_position[0], text_position[1] - text_size[1] - 5),
                            (text_position[0] + text_size[0] + 5, text_position[1] + 5),
                            (0, 0, 0),  # Black background
                            -1
                        )
                        cv2.putText(output_image, angle_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)




        # Display the image
        cv2.imshow("Image with Points, Lines, and Equations", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def find_slope_and_intercept(X, y):
        """
        Perform linear regression for given X and y.

        Parameters:
            X: List or numpy array of x-coordinates.
            y: List or numpy array of y-coordinates.

        Returns:
            params: Tuple (slope, intercept)
                    - For standard regression: (slope, intercept)
                    - For vertical lines: (np.inf, constant_x_value)
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        if len(set(X)) == 1:  # Check if all X-values are identical
            return np.inf, X[0]

        # Add bias term for linear regression
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        try:
            params = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
        except np.linalg.LinAlgError:
            raise ValueError("Cannot compute linear regression; check input data.")

        return params[1], params[0]

    def point_to_point(self, point1, point2):
        """
        Calculate the Euclidean distance between two points.

        Args:
            point1 (tuple): Coordinates of the first point (x1, y1, [z1]).
            point2 (tuple): Coordinates of the second point (x2, y2, [z2]).

        Returns:
            float: Distance between the two points.
        """
        if len(point1) != len(point2):
            raise ValueError("Both points must have the same dimension.")
        self.lines[(point1, point2)] = (math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2))))
        self.add_point(point1)
        self.add_point(point2)
        return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

    def point_to_line_segment(self, point, line):
        """
        Calculates the minimum distance and angle between a point and a line segment.

        Parameters:
            point: Tuple (x0, y0) - Coordinates of the point.
            line: Tuple ((x1, y1), (x2, y2)) - Endpoints of the line segment.

        Returns:
            distance: Minimum distance from the point to the line segment.
            angle: Angle (in degrees) between the vector to the closest point and the line segment.
            nearest_point: Coordinates of the nearest point on the line segment.
        """
        line_start, line_end = map(np.array, line)
        point = np.array(point)

        line_vector = line_end - line_start
        line_length_squared = np.dot(line_vector, line_vector)

        if np.isclose(line_length_squared, 0):
            raise ValueError("The two endpoints of the segment cannot be the same.")

        # Projection factor
        t = np.dot(point - line_start, line_vector) / line_length_squared
        t = np.clip(t, 0, 1)  # Clamp t to [0, 1]

        projection = line_start + t * line_vector
        distance = np.linalg.norm(point - projection)
        
        vector_to_point = projection - point
        vector_to_point_magnitude = np.linalg.norm(vector_to_point)
        line_vector_magnitude = np.linalg.norm(line_vector)

        if np.isclose(vector_to_point_magnitude, 0) or np.isclose(line_vector_magnitude, 0):
            angle = 0.0
        else:
            cosine_angle = np.dot(vector_to_point, line_vector) / (vector_to_point_magnitude * line_vector_magnitude)
            angle = math.degrees(math.acos(np.clip(cosine_angle, -1, 1)))

        self.lines[tuple(map(tuple, (point, projection)))] = distance
        self.lines[tuple(map(tuple, (line_start, line_end)))] = None
        self.add_point(tuple(projection))
        return distance, angle, tuple(projection)

    def point_to_line(self, point, line):
        """
        Calculates the minimum distance and angle between a point and an infinite line.

        Parameters:
            point: Tuple (x0, y0) - Coordinates of the point.
            line: Tuple (slope, intercept) - Slope and intercept of the infinite line.

        Returns:
            distance: Minimum distance from the point to the line.
            angle: Angle (in degrees) between the vector to the closest point and the line.
            nearest_point: Coordinates of the nearest point on the line.
        """
        slope, intercept = line
        x0, y0 = point

        if math.isinf(slope):  # Vertical line
            projection_x = intercept
            projection_y = y0
        else:
            perpendicular_slope = -1 / slope
            perpendicular_intercept = y0 - perpendicular_slope * x0

            projection_x = (perpendicular_intercept - intercept) / (slope - perpendicular_slope)
            projection_y = slope * projection_x + intercept

        distance = math.sqrt((x0 - projection_x) ** 2 + (y0 - projection_y) ** 2)

        vector_to_point = (projection_x - x0, projection_y - y0)
        line_vector = (1, slope)

        dot_product = vector_to_point[0] * line_vector[0] + vector_to_point[1] * line_vector[1]
        vector_to_point_magnitude = math.sqrt(vector_to_point[0] ** 2 + vector_to_point[1] ** 2)
        line_vector_magnitude = math.sqrt(line_vector[0] ** 2 + line_vector[1] ** 2)

        if np.isclose(vector_to_point_magnitude, 0) or np.isclose(line_vector_magnitude, 0):
            angle = 0.0
        else:
            cosine_angle = dot_product / (vector_to_point_magnitude * line_vector_magnitude)
            angle = math.degrees(math.acos(np.clip(cosine_angle, -1, 1)))

        self.lines[(tuple((projection_x, projection_y)), point)] = distance
        self.line_equations[(line, None)] = None     
        self.add_point((projection_x, projection_y))
        return distance, angle, (projection_x, projection_y)

    def line_to_line(self, line1, line2):
        """
        Calculate the minimum distance between two line segments in 2D space
        and return the points on the segments where this distance occurs.

        Parameters:
            line1: Tuple of two points defining the first line segment ((x1, y1), (x2, y2)).
            line2: Tuple of two points defining the second line segment ((x3, y3), (x4, y4)).

        Returns:
            distance: Minimum distance between the two line segments.
            point_on_line1: Closest point on the first segment.
            point_on_line2: Closest point on the second segment.
        """    
        p1, q1 = map(np.array, line1)
        p2, q2 = map(np.array, line2)

        d1 = q1 - p1  # Direction vector of line1
        d2 = q2 - p2  # Direction vector of line2

        cross_d1_d2 = np.cross(d1, d2)

        if np.isclose(cross_d1_d2, 0):  # Parallel lines
            # Project p2 onto line1 and clamp to segment
            t = np.dot(p2 - p1, d1) / np.dot(d1, d1)
            t = max(0, min(1, t))  # Clamp t to [0, 1]
            point_on_line1 = p1 + t * d1

            # Find the closest point from line2's endpoints to line1
            distances = [
                (np.linalg.norm(point_on_line1 - p2), p2),
                (np.linalg.norm(point_on_line1 - q2), q2),
            ]
            point_on_line2 = min(distances, key=lambda x: x[0])[1]

            distance = np.linalg.norm(point_on_line1 - point_on_line2)
            self.lines[(tuple(point_on_line1), tuple(point_on_line2))] = distance
            self.lines[(tuple(p1),tuple(q1))] = None
            self.lines[(tuple(p2),tuple(q2))] = None
                
            self.add_point(tuple(point_on_line1))
            self.add_point(tuple(point_on_line2))
            return distance, tuple(point_on_line1), tuple(point_on_line2)

        # Non-parallel lines
        t1 = np.cross((p2 - p1), d2) / cross_d1_d2
        t2 = np.cross((p2 - p1), d1) / cross_d1_d2

        # Clamp t1 and t2 to [0, 1] to restrict to segments
        t1 = max(0, min(1, t1))
        t2 = max(0, min(1, t2))

        # Compute the closest points on the segments
        point_on_line1 = p1 + t1 * d1
        point_on_line2 = p2 + t2 * d2

        distance = np.linalg.norm(point_on_line1 - point_on_line2)
        
        self.lines[(tuple(point_on_line1), tuple(point_on_line2))] = distance
        self.lines[(tuple(p1),tuple(q1))] = None
        self.lines[(tuple(p2),tuple(q2))] = None
               
        self.add_point(tuple(point_on_line1))
        self.add_point(tuple(point_on_line2))


        return distance, tuple(point_on_line1), tuple(point_on_line2)




    def find_angle_and_intersection(self, line1, line2):
        """
        Calculate the angle (in degrees) between two lines given their slopes and intercepts,
        and find the point of intersection of the lines.

        Parameters:
            line1 (tuple): (slope1, intercept1) of the first line.
            line2 (tuple): (slope2, intercept2) of the second line.

        Returns:
            tuple: (angle_in_degrees, intersection_point)
                - angle_in_degrees (float): Angle between the two lines in degrees.
                - intersection_point (tuple): (x, y) coordinates of the intersection point, or None if lines are parallel.
        """
        if (line1, line2) in self.line_equations or (line2, line1) in self.line_equations:
            return self.line_equations.get((line1, line2)) or self.line_equations.get((line2, line1))

        slope1, intercept1 = line1
        slope2, intercept2 = line2

        # Check for parallel lines
        if np.isclose(slope1, slope2):  # Parallel lines
            self.line_equations[(line1, line2)] = None
            return 0, None

        # Handle vertical and horizontal lines
        if math.isinf(slope1) and slope2 == 0:  # Vertical and horizontal
            x_intersection = intercept1  # x = c
            y_intersection = intercept2  # y = k
            intersection_point = (x_intersection, y_intersection)
            angle = 90.0
            self.line_equations[(line1, line2)] = intersection_point
            return angle, intersection_point

        if math.isinf(slope2) and slope1 == 0:  # Horizontal and vertical
            x_intersection = intercept2  # x = c
            y_intersection = intercept1  # y = k
            intersection_point = (x_intersection, y_intersection)
            angle = 90.0
            self.line_equations[(line1, line2)] = (intersection_point,angle)
            return angle, intersection_point

        # General case: Find intersection point
        try:
            x_intersection = (intercept2 - intercept1) / (slope1 - slope2)
            y_intersection = slope1 * x_intersection + intercept1
            intersection_point = (x_intersection, y_intersection)
        except ZeroDivisionError:
            return 0, None  # Handle cases where the calculation fails (shouldn't happen here)

        # Calculate angle
        try:
            if np.isclose(slope1 * slope2, -1):  # Perpendicular lines
                angle = 90.0
            else:
                angle = math.degrees(math.atan(abs((slope2 - slope1) / (1 + slope1 * slope2))))
        except ZeroDivisionError:
            angle = 90.0  # Handle perpendicular cases explicitly

        self.line_equations[(line1, line2)] = (intersection_point,angle)
        return angle, intersection_point
# image=cv2.imread('fick.jpg')
# measure=Measurement(image)
# point=(237.38608554960058, 122.20912124582874)
# point2=(86.90737182728286, 189.20386287794523)
# point3=(158.01233694003443, 370.93770856507234)
# measure.add_point(point)
# measure.add_point(point2)
# measure.add_point(point3)
# measure.draw()