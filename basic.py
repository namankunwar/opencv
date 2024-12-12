import cv2 as cv

img = cv.imread("photos/blank.jpg")

resize_img = cv.resize(img, (500,500), interpolation=cv.INTER_AREA )

#converting to gray scale
gray = cv.cvtColor(resize_img,cv.COLOR_BGR2GRAY)

#blur
blur = cv.GaussianBlur(resize_img, (5,5), cv.BORDER_DEFAULT)

#edge cascade
canny = cv.Canny(resize_img, 125, 150 )

#dilating image
dilate = cv.dilate(canny, (3,3), iterations=3)

eroded= cv.erode(dilate, (3,3), iterations=1)

#cropping
crop = resize_img[200:300, 300:700]


cv.imshow("erodded", eroded)
cv.imshow("cropped", crop)
cv.imshow("gray img", gray)
cv.imshow("dilate img", dilate)
cv.imshow("edge cascade", canny)
cv.imshow("blur img", blur)
cv.imshow("original image", resize_img)
cv.waitKey(0)
