import cv2 as cv
import numpy as np
import math

def draw_lines(image, start_coords, length1, angle1, angle2):
    height, width = image.shape[:2]

    angle1_rad = math.radians(angle1)
    angle2_rad = math.radians(angle2)

    # endpoint of the first line
    x1 = int(start_coords[0] + length1 * math.cos(angle1_rad))
    y1 = int(start_coords[1] - length1 * math.sin(angle1_rad))

    # slope and intercept of the second line
    total_angle = angle1_rad + angle2_rad
    slope = -math.tan(total_angle) 
    intercept = y1 - slope * x1  # c = y - mx

    # Find intersection points with image boundaries
    points = []

    # Left boundary (x = 0)
    y_left = slope * 0 + intercept
    if 0 <= y_left < height:
        points.append((0, int(y_left)))

    # Right boundary (x = width)
    y_right = slope * width + intercept
    if 0 <= y_right < height:
        points.append((width, int(y_right)))

    # Top boundary (y = 0)
    if slope != 0:  # Avoid division by zero
        x_top = (0 - intercept) / slope
        if 0 <= x_top < width:
            points.append((int(x_top), 0))

    # Bottom boundary (y = height)
    if slope != 0:  # Avoid division by zero
        x_bottom = (height - intercept) / slope
        if 0 <= x_bottom < width:
            points.append((int(x_bottom), height))


    if len(points) >= 2:
     
        cv.line(image, points[0], points[1], (255, 0, 0), 2)  # scond line

        cv.circle(image, start_coords, 2, (0, 255, 0), -1)  # Starting point


    return image

image = np.zeros((500, 500, 3), dtype=np.uint8)

result_image = draw_lines(image, start_coords= (100, 300)  , length1 = 150, angle1= 45  , angle2= 120 )

cv.imshow("Connected Lines with Boundary", result_image)
cv.waitKey(0)
cv.destroyAllWindows()
