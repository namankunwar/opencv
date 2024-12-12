import cv2 as cv

# Load the BGR image
rimg= cv.imread("photos/blank.jpg")

img = cv.resize(rimg, (500,500), interpolation=cv.INTER_AREA )

#Averging
a_blur = cv.blur(img, (3,3), cv.BORDER_DEFAULT)

#Gaussian blur
g_blur = cv.GaussianBlur(img, (3,3),0, cv.BORDER_DEFAULT)

#median blur
m_blur= cv.medianBlur(img, 3)

#Bialateral blur
b_blur= cv.bilateralFilter(img, 10, 25, 20)

cv.imshow("median blur image", m_blur)
cv.imshow("Bialateral blur image", b_blur)
cv.imshow("original image", img)
cv.imshow("Average blur image", a_blur)
cv.imshow("Gaussian blur image", g_blur)
cv.waitKey(0)