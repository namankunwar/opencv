import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load exposure images
# Provide a list of images captured at different exposure levels
# Ensure the images are of the same scene with different exposure values
exposure_images = [
    cv2.imread("photos/low_exposure.jpg"),
    cv2.imread("photos/medium_exposure.jpg"),
    cv2.imread("photos/high_exposure.jpg")
]

# Ensure all images are loaded
if any(img is None for img in exposure_images):
    raise ValueError("One or more images couldn't be loaded. Check file paths.")

# Convert images to the same size for consistency (optional)
# Useful when images might slightly vary in dimensions due to cropping or resizing
size = (600, 400)  # Width x Height
exposure_images = [cv2.resize(img, size) for img in exposure_images]

# Step 2: Define exposure times for each image
# These times represent the exposure duration (in seconds) used to capture each image
# Shorter exposure captures less light, longer captures more
exposure_times = np.array([1/30.0, 1/60.0, 1/250.0], dtype=np.float32)  # Example values

# Step 3: Create a CalibrateDebevec object for HDR calibration
# cv2.createCalibrateDebevec() creates a model to estimate camera response
calibrate = cv2.createCalibrateDebevec()
response_debevec = calibrate.process(exposure_images, exposure_times)

# Step 4: Merge the images into an HDR image
# Use cv2.createMergeDebevec() for merging
merge_debevec = cv2.createMergeDebevec()
hdr_image = merge_debevec.process(exposure_images, exposure_times, response_debevec)

# Step 5: Tonemap the HDR image to make it viewable on standard displays
# HDR images have a high range of intensities; tonemapping maps them to a displayable range
tonemap = cv2.createTonemap(gamma=2.2)  # Gamma correction (default is 2.2 for typical displays)
ldr_image = tonemap.process(hdr_image)

# Scale the LDR image to [0, 255] for saving and display
ldr_image = np.clip(ldr_image * 255, 0, 255).astype('uint8')

# Step 6: Display results
# Displaying the LDR image after tonemapping
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(ldr_image, cv2.COLOR_BGR2RGB))  # Convert to RGB for proper color display
plt.title("Tonemapped HDR Image")
plt.axis('off')
plt.show()

# Optional: Save the tonemapped image
cv2.imwrite("photos/tonemapped_hdr.jpg", ldr_image)

# Parameters and Explanations
# ----------------------------------------------
# 1. Exposure times:
#    - Array of floats representing exposure times for each image.
#    - Must match the number of images in `exposure_images`.
#    - Values depend on the camera settings used to capture images.

# 2. Gamma (cv2.createTonemap):
#    - Controls brightness and contrast in the tonemapped image.
#    - Lower values (<2.2) give darker images; higher values (>2.2) give brighter images.
#    - Adjust based on your display requirements.

# 3. cv2.createCalibrateDebevec():
#    - Computes camera response function based on the exposure times and images.
#    - Parameters can include options for advanced calibration, but the defaults are generally good.

# 4. cv2.createMergeDebevec():
#    - Combines multiple exposure images into an HDR image.
#    - Requires consistent images and exposure times.

# 5. cv2.createTonemap():
#    - Maps the high dynamic range of the HDR image to a viewable range.
#    - Gamma correction adjusts the intensity of the resulting image.

# Key Notes:
# - Ensure consistent alignment of input images. Misaligned images can cause artifacts.
# - Use a tripod and a fixed scene while capturing images to ensure uniformity.
# - Experiment with different gamma values for tonemapping to achieve the desired look.
