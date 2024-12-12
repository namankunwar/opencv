import cv2 as cv
import numpy as np

img= cv.imread("photos/blank.jpg")

resize_img = cv.resize(img, (500,500), interpolation=cv.INTER_AREA )

#translation

def translate(img, x, y):
    transmat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transmat, dimensions)

# -x -> left
#-y -> up
# x -> right
# y -> down

translated = translate(resize_img, -50, 80)

#rotation 
def rotate(img, angle, rotPoint= None):
    (height, width)= img.shape[:2]

    if rotPoint is None:
        rotPoint = (width //2, height // 2)
    
    rotmat = cv.getRotationMatrix2D(rotPoint,angle, 1)
    dimension = (width, height)
    return cv.warpAffine(img, rotmat, dimension)

rotated = rotate(resize_img,45)


#flipping

img_flip = cv.flip(resize_img, 1)
cv.imshow("flipping", img_flip)

cv.imshow("tranlate" , translated)
cv.imshow("rotated" , rotated)
cv.imshow("orginal image", resize_img)
cv.waitKey(0)

