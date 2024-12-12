import cv2 as cv
import numpy as np

#img = cv.imread("photos/anime.jpg")
#cv.imshow("bv", img)

#creating a blank image to draw
dummy = np.zeros((500,500,3), dtype="uint8")

#paint the image 
#dummy[:]=0,0,255 #bgr, so it shows red color only. ":" this means all pixels
#dummy[200:300, 300:700]= 0,0,255 # same but only range of pixels is colored red
#cv.imshow("dum",dummy)

#draw a rectangle
#cv.rectangle(dummy, (0,100),(300,400), (255,0,0), thickness=3) #WE CAN WRITE cv.fill or -1 in thickness to fill the portion

cv.rectangle(dummy, (0,100),(dummy.shape[1]//2, dummy.shape[0]//2),(255,0,0), thickness= cv.FILLED) 
cv.imshow("dum",dummy)

#draw a circle
cv.circle(dummy, (150,150),30, (0,0,255), thickness= 3)
cv.imshow("dum", dummy)

#draw a white line
cv.line(dummy, (150,150) ,(300,300), (255,255,255), thickness= 3)
cv.imshow("dum", dummy)


#write text in image
cv.putText(dummy, text="hello bro !",org=(0,100), fontFace=1, fontScale=3, color=(0,124,230), thickness=2)
cv.imshow("dum",dummy)

cv.waitKey(0)