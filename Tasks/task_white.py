
import cv2 as cv
import numpy as np

# Input array
array1 = [25, 26, 27, 28, 21, 14, 7, 6, 5, 4, 3, 2, 1, 8, 15, 22, 23, 24]
blank1 = np.zeros((4, 7)).reshape(-1)
array2 = np.array(array1) - 1


# animation
for index in array2:
    blank1[index] = 1  
    blank1_reshaped = blank1.reshape(4, 7) 
    img1 = cv.resize(blank1_reshaped, (400, 700), interpolation=cv.INTER_NEAREST)  
    cv.imshow("Animations", img1)  
    cv.waitKey(200)  


cv.destroyAllWindows()
