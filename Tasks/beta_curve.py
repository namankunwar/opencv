import cv2
import numpy as np

def classify_logarithmic_L_shape(contour):
  
    # Extract and sort points
    points = np.squeeze(contour)  # Remove unnecessary dimensions
    points = np.array(points)  # Ensure points are in array form

    #  Find x_max and x_min points
    x_max_point = points[np.argmax(points[:, 0])]  # Point with max x
    x_min_point = points[np.argmin(points[:, 0])]  # Point with min x

    # Check y-value trends for x_max
    x_max_index = np.argmax(points[:, 0])
    x_max_y_trend = "increasing" if points[x_max_index - 1, 1] < points[x_max_index, 1] else "decreasing"

    #Check y-value trends for x_min
    x_min_index = np.argmin(points[:, 0])
    x_min_y_trend = "increasing" if points[x_min_index - 1, 1] < points[x_min_index, 1] else "decreasing"

    # Classify curve
    if x_max_y_trend == "increasing":
        curve_type = "Bottom-left L-shape (L)"
    elif x_max_y_trend == "decreasing":
        curve_type = "Top-left L-shape (Γ)"
    elif x_min_y_trend == "increasing":
        curve_type = "Bottom-right flipped L (⅃)"
    elif x_min_y_trend == "decreasing":
        curve_type = "Top-right flipped Γ (flipped Γ)"
    else:
        curve_type = "Unknown shape"

    return curve_type



# Create a test image with a logarithmic L-curve
image = np.zeros((500, 500), dtype=np.uint8)
cv2.line(image, (50, 250), (250, 250), 255, 5)  # Horizontal segment
cv2.line(image, (250, 250), (250, 450), 255, 5)  # Vertical segment

# Find contours
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#Process each contour
for contour in contours:
    curve_type = classify_logarithmic_L_shape(contour)
    print(f"Curve Type: {curve_type}")
  
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
   
    cv2.imshow("Classified Curve", color_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
