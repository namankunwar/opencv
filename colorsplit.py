import cv2 as cv
import matplotlib.pyplot as plt

# Load the BGR image
rimg= cv.imread("photos/blank.jpg")

img = cv.resize(rimg, (500,500), interpolation=cv.INTER_AREA )
# Convert the BGR image to RGB (since OpenCV loads in BGR by default)
img_org = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert BGR to RGB for correct display

# Split the image into Blue, Green, and Red channels
b, g, r = cv.split(img)  # This splits the image into three separate channels: Blue, Green, and Red

# Create a figure to display the three channels
plt.figure(figsize=(20, 5))  # Set the figure size to display the images

# Display the original
plt.subplot(1, 5, 1)  # Create a subplot (1 row, 3 columns, position 1)
plt.imshow(img_org)  # Show the original
plt.title("orignal ")  # Title for this subplot
plt.axis("off")  # Turn off the axis

#lighter color means more concentration of the concern color here in first example "blue"

# Display the Blue channel
plt.subplot(1, 5, 2)  # Create a subplot (1 row, 3 columns, position 1)
plt.imshow(b, cmap='gray')  # Show the Blue channel using the 'Blues' colormap
plt.title("Blue Channel")  # Title for this subplot
plt.axis("off")  # Turn off the axis

# Display the Green channel
plt.subplot(1, 5, 3)  # Create a subplot (1 row, 3 columns, position 2)
plt.imshow(g, cmap='gray')  # Show the Green channel using the 'Greens' colormap
plt.title("Green Channel")  # Title for this subplot
plt.axis("off")  # Turn off the axis

# Display the Red channel
plt.subplot(1, 5, 4)  # Create a subplot (1 row, 3 columns, position 3)
plt.imshow(r, cmap='gray')  # Show the Red channel using the 'Reds' colormap
plt.title("Red Channel")  # Title for this subplot
plt.axis("off")  # Turn off the axis


# Merge the channels back into a single image (BGR format)
merged_img = cv.merge((b, g, r))  # Merging Blue, Green, Red channels

# Convert the merged image from BGR to RGB for displaying in matplotlib (OpenCV uses BGR by default)
merged_img_rgb = cv.cvtColor(merged_img, cv.COLOR_BGR2RGB)

# Display the merged image
plt.subplot(155)
plt.imshow(merged_img_rgb)  # Display the merged image (converted to RGB)
plt.title("Merged Image")

# Show the plots
plt.show()  # Display the three color channels