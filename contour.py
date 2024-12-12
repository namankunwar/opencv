import cv2 as cv
import numpy as np
img= cv.imread("photos/blank.jpg")

resize_img = cv.resize(img, (500,500), interpolation=cv.INTER_AREA )

blank = np.zeros(resize_img.shape, dtype="uint8")

blur= cv.GaussianBlur(resize_img, (9,9), cv.BORDER_DEFAULT)

canny = cv.Canny(blur, 135, 155)

# Load image in grayscale
gray_img = cv.cvtColor(resize_img, cv.COLOR_BGR2GRAY)

# Apply binary threshold to convert the image into a binary image
#ret, threshold = cv.threshold(gray_img, 180, 255, cv.THRESH_BINARY)

# Find contours in the binary image
contour, heirachy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(f"{len(contour)} founded")

#cv.imshow("canny", canny)
cv.drawContours(resize_img, contour, -1, (0,255,0),1)  # Draw contours in green in original image

cv.drawContours(blank, contour, -1, (0,255,0),1)  # Draw contours in green in blank image
cv.imshow("original", resize_img)
cv.imshow("gray", gray_img)
cv.imshow("blank", blank)
cv.waitKey(0)