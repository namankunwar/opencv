import cv2  # OpenCV for computer vision tasks.
import sys  # To handle command-line arguments.
import numpy  # For numerical operations and array handling.

# Define modes for image processing filters
PREVIEW = 0  # Preview mode: shows the original frame.
BLUR = 1  # Blur mode: applies a simple blurring filter.
FEATURES = 2  # Features mode: detects corner features.
CANNY = 3  # Canny mode: applies the Canny edge detection.

# Parameters for the corner detection (goodFeaturesToTrack)
feature_params = dict(
    maxCorners=500,  # Maximum number of corners to detect.
    qualityLevel=0.2,  # Minimum accepted quality of corners (0.0 to 1.0).
    minDistance=15,  # Minimum distance between corners.
    blockSize=9  # Size of the block used for computing corner detection.
)

# Default video source (camera index 0 or video file path passed as an argument)
#s = 0
#if len(sys.argv) > 1:  # Check if a video source is passed as a command-line argument.
 #   s = sys.argv[1]  # Use the provided argument as the source.

# Ask the user to input the video file path
video_path = input("Enter the video file path: ") 


# Initialize the filter mode and control variables
image_filter = PREVIEW  # Start with the default mode (Preview).
alive = True  # Control variable for the while loop.

# Create a window for displaying the output
win_name = "Camera Filters"  # Name of the OpenCV window.
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)  # Create a resizable window.

# Initialize the video capture source
#source = cv2.VideoCapture(s)  # Open the camera or video file specified by `s`.

# Open the video file
source = cv2.VideoCapture(video_path)  # Open the specified video file

if not source.isOpened():  # Check if the video file can be opened
    print("Error: Cannot open the video file.")
    sys.exit()

# Get the video's frame rate
fps = source.get(cv2.CAP_PROP_FPS)  # Frames per second
frame_delay = int(1000 / fps)  # Delay in milliseconds between frames


while alive:  # Infinite loop to process frames until the user exits.
    has_frame, frame = source.read()  # Read the next frame from the video source.
    if not has_frame:  # If no frame is retrieved (e.g., end of video), break the loop.
        break

    #frame = cv2.flip(frame, 1)  # Flip the frame horizontally (mirror effect).

    # Apply the selected filter to the frame
    if image_filter == PREVIEW:
        result = frame  # Show the original frame without processing.
    elif image_filter == CANNY:
        # Apply Canny edge detection.
        # Threshold1 (80) and Threshold2 (150) determine edge detection sensitivity.
        result = cv2.Canny(frame, 80, 150)
    elif image_filter == BLUR:
        # Apply a blur filter with a kernel size of 13x13.
        # Larger kernel size results in more smoothing.
        result = cv2.blur(frame, (13, 13))
    elif image_filter == FEATURES:
        result = frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
        corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)  # Detect corners
        if corners is not None:  # If corners are detected
            for x, y in numpy.float32(corners).reshape(-1, 2):  # Iterate through corner points
                cv2.circle(result, (int(x), int(y)), 10, (0, 255, 0), 1)  # Draw circles

    # Display the processed frame in the OpenCV window.
    cv2.imshow(win_name, result)

    # Listen for keypress events to control the application
    key = cv2.waitKey(frame_delay)  # Wait for 1 ms for a key press.
    if key == ord("Q") or key == ord("q") or key == 27:  # Exit on 'Q', 'q', or 'Esc'.
        alive = False  # Break the loop and exit.
    elif key == ord("C") or key == ord("c"):  # Switch to Canny filter on 'C' or 'c'.
        image_filter = CANNY
    elif key == ord("B") or key == ord("b"):  # Switch to Blur filter on 'B' or 'b'.
        image_filter = BLUR
    elif key == ord("F") or key == ord("f"):  # Switch to Features filter on 'F' or 'f'.
        image_filter = FEATURES
    elif key == ord("P") or key == ord("p"):  # Switch to Preview mode on 'P' or 'p'.
        image_filter = PREVIEW

# Release resources and clean up
source.release()  # Release the video capture source.
cv2.destroyWindow(win_name)  # Close the OpenCV window.
