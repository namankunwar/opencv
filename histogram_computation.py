import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("photos/butterfly.jpg")  # Replace with the path to your image


# Compute BGR Histogram
colors_bgr = ('b', 'g', 'r')  # Color channels
plt.figure(figsize=(10, 5))  # Create a figure for BGR
plt.title("BGR Histogram")
plt.xlabel("Intensity")
plt.ylabel("Count")
for i, color in enumerate(colors_bgr):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(hist, color=color, label=f'{color.upper()} Channel')  # Plot histogram
    plt.xlim([0, 256])
plt.legend()
plt.show()

# Convert to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Compute RGB Histogram
colors_rgb = ('r', 'g', 'b')  # RGB channels
plt.figure(figsize=(10, 5))  # Create a figure for RGB
plt.title("RGB Histogram")
plt.xlabel("Intensity")
plt.ylabel("Count")
for i, color in enumerate(colors_rgb):
    hist = cv2.calcHist([rgb_image], [i], None, [256], [0, 256])
    plt.plot(hist, color=color, label=f'{color.upper()} Channel')  # Plot histogram
    plt.xlim([0, 256])
plt.legend()
plt.show()
