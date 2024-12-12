import cv2 as cv
import matplotlib.pyplot as plt

# Load the BGR image
rimg= cv.imread("photos/blank.jpg")

img = cv.resize(rimg, (500,500), interpolation=cv.INTER_AREA )

# Convert the BGR image to RGB (since OpenCV loads in BGR by default)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert BGR to RGB for correct display

# Convert the BGR image to HSV (Hue, Saturation, Value color space)
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # Converts from BGR to HSV

# Convert the BGR image to Grayscale (a black-and-white image)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert to Grayscale

# Convert the BGR image to LAB (Lightness, A, B color space)
img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)  # Converts from BGR to LAB color space

# Create a figure with a specific size to display the images
plt.figure(figsize=(20, 10))  # Set the size of the figure (width, height)

# Display the original RGB image
plt.subplot(2, 3, 1)  # Create a subplot (2 rows, 3 columns, position 1)
plt.imshow(img_rgb)  # Show the RGB image
plt.title("RGB Image")  # Title for this subplot
plt.axis("off")  # Turn off the axis for a cleaner image display

# Display the original RGB image
plt.subplot(2, 3, 1)  # Create a subplot (2 rows, 3 columns, position 1)
plt.imshow(img_rgb)  # Show the RGB image
plt.title("RGB Image")  # Title for this subplot
plt.axis("off")  # Turn off the axis for a cleaner image display

# Display the HSV image
plt.subplot(2, 3, 2)  # Create a subplot (2 rows, 3 columns, position 2)
plt.imshow(img_hsv)  # Show the HSV image
plt.title("HSV Image")  # Title for this subplot
plt.axis("off")  # Turn off the axis

# Display the Grayscale image
plt.subplot(2, 3, 3)  # Create a subplot (2 rows, 3 columns, position 3)
plt.imshow(img_gray, cmap='gray')  # Show the grayscale image using the 'gray' colormap
plt.title("Grayscale Image")  # Title for this subplot
plt.axis("off")  # Turn off the axis

# Display the LAB image (converted back to RGB for visualization)
plt.subplot(2, 3, 4)  # Create a subplot (2 rows, 3 columns, position 4)
plt.imshow(cv.cvtColor(img_lab, cv.COLOR_LAB2RGB))  # Convert LAB back to RGB for display
plt.title("LAB Image")  # Title for this subplot
plt.axis("off")  # Turn off the axis

# Show the plots (all images will be displayed in one window)
plt.show()  # Display the images