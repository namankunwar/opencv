import cv2  # Import the OpenCV library for image and video processing.
import sys  # Import the sys module to handle command-line arguments.

# Initialize the video source index or path
s = 0  # Default video source index (usually the first webcam on the system).
if len(sys.argv) > 1:  # Check if additional arguments are passed via the command line.
    s = sys.argv[1]  # If so, use the first argument as the video source (file path or index).

# Open the video capture source
source = cv2.VideoCapture(s)  # Create a VideoCapture object for the given source.
# `s` can be:
# - An integer: To use a webcam or camera device (e.g., 0 for default webcam, 1 for the second camera).
# - A string: Path to a video file.

# Create a named window for displaying the video
win_name = 'Camera Preview'  # Name of the window for displaying video frames.
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)  # Create a resizable OpenCV window.

# Start a loop to display 
# the video feed
while cv2.waitKey(1) != 27:  # Wait for 1 ms between frames and check for the 'Esc' key (ASCII code 27).
    has_frame, frame = source.read()  # Read a frame from the video source.
    # `has_frame`: Boolean indicating if a frame was successfully read.
    # `frame`: The actual video frame (image).

    if not has_frame:  # If no frame is available (e.g., end of video or camera disconnected).
        break  # Exit the loop.

    cv2.imshow(win_name, frame)  # Display the frame in the window.

# Release resources
source.release()  # Release the VideoCapture object and free the video source.
cv2.destroyWindow(win_name)  # Close the OpenCV window.
