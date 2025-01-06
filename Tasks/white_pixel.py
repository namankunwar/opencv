
import cv2 as cv
import numpy as np

# Input array
array1=[25,26,27,28,21,14,7,6,5,4,3,2,1,8,15,22,23,24]
blank1 = np.zeros((4,7), dtype="uint8")

#Taking cordinates from the array1 where 7 is image width
cordinates = (( (index - 1) // 7, (index - 1) % 7) for index in array1)

# animation
for row, col in cordinates:
    blank1[row,col]=255

    img1= cv.resize(blank1,(400,700), interpolation=cv.INTER_NEAREST)

    cv.imshow("Animation", img1)
    
    cv.waitKey(200)

cv.destroyAllWindows()
