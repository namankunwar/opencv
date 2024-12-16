import cv2 as cv


image =cv.imread("photos/good-image.jpg")

if image is None:
        print(f"Error loading image: {image}")
        
# Calculate metrics

gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

"""Calculate the mean brightness of an image."""
brightness = gray_image.mean()

"""Calculate sharpness using Laplacian variance."""
laplacian  = cv.Laplacian(gray_image, cv.CV_64F) # Apply Laplacian filter
sharpness = laplacian.var()  # Calculate variance

brightness_thresh = 60
sharpness_thresh = 500

print(f"Brightness: {brightness:.2f}, Sharpness: {sharpness:.2f}")

    # Classify image based on thresholds
if brightness < brightness_thresh:
    print("Bad - Too Dark")
elif sharpness < sharpness_thresh:
    print("Bad - Too blurry")
else:
    print("good image")


