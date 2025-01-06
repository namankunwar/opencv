import cv2 as cv
import numpy as np

# Load the apple image
apple_img = cv.imread("apple_image.jpg")  
apple_img = cv.resize(apple_img, (512, 512), interpolation=cv.INTER_AREA)  

# Create a blank image 
blank_image = np.zeros((4 * 512, 7 * 512, 3), dtype=np.uint8)

#sequence of the apple image positions in the 4x7 grid 
array1 = [25, 26, 27, 28, 21, 14, 7, 6, 5, 4, 3, 2, 1, 8, 15, 22, 23, 24]

pixel_size = 512  # Size of each apple image (512x512)

#Taking cordinates from the array1 where 7 is image width
cordinates = (( ((index  - 1) // 7)*pixel_size, ((index - 1) % 7)*pixel_size) for index in array1)

# sequential order
for row, col in cordinates:
    
    # Place the apple image at the calculated position (Slicing)
    blank_image[row:row + pixel_size, col:col + pixel_size] = apple_img
    
    #resize the blank_image to make it fit
    resized_image = cv.resize(blank_image,(512,512), interpolation=cv.INTER_AREA)
    
    cv.imshow("Animation", resized_image)
    
    cv.waitKey(200) 


cv.destroyAllWindows()
