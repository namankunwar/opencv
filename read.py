import cv2 as cv

'''
img = cv.imread("photo/butterfly.jpg")
cv.imshow('Butterfly', img)
cv.waitKey(5000)

img2 = cv.imread("photo/building.jpg")
cv.imshow('Building', img2)
cv.waitKey(0)
'''

# Video Capture Section
vd = cv.VideoCapture("video/sample.mp4")  # Use 0 or 1 for webcam or camera inputs

while True:
    isTrue, frame = vd.read()
    
    # Break the loop if the video ends or there's an error in reading
    if not isTrue:
        print("Video ended or error in reading the frame.")
        break
    
    cv.imshow('Video', frame)
    
    key = cv.waitKey(30) & 0xFF
    
    if key == ord("s"):  # Exit on "s"
        print("Exiting video playback on 's' key.")
        break
    if cv.getWindowProperty('Video', cv.WND_PROP_VISIBLE) < 1:  # If the user closes the window, the function returns a value < 1
        print("Window closed.")
        break

# Release resources and close windows
vd.release()
cv.destroyAllWindows()
