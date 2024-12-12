import cv2 as cv

vd = cv.VideoCapture("video/sample.mp4")

# Check if video file was opened successfully
if not vd.isOpened():
    print("Error: Unable to open video file.")
    exit()

def rescale_video(frame, scale):
    """
    Rescales the input video frame by the given scale factor.
    :param frame: Input video frame
    :param scale: Scale factor (e.g., 0.5 for half size, 2.0 for double size)
    :return: Rescaled video frame
    """
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimension = (width, height)
    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)

while True:
        isTrue, frame = vd.read()
        if not isTrue:
            print("Video ended or error in reading the frame.")
            break

        rescale_vd = rescale_video(frame, 0.5)

        cv.imshow("original video", frame)
        cv.imshow("rescale video", rescale_vd)

        key = cv.waitKey(30) & 0xFF
        if key == ord("s"):
            print("Exiting video playback on 's' key.")
            break
        if cv.getWindowProperty('original video', cv.WND_PROP_VISIBLE) < 1: # If the user closes the window, the function returns a value < 1
            print("Window closed.")
            break


# Release resources and close windows
vd.release()
cv.destroyAllWindows()

