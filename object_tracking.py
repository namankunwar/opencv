import cv2
import sys

# List of available trackers
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
tracker_type = 'MIL'  # Select the tracker you want to use (KCF is a good general-purpose tracker)

# Select the tracker using the OpenCV API
if tracker_type == 'BOOSTING':
    tracker = cv2.TrackerBoosting_create()
elif tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create() 
elif tracker_type == 'TLD':
    tracker = cv2.TrackerTLD_create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv2.TrackerMedianFlow_create()
elif tracker_type == 'MOSSE':
    tracker = cv2.TrackerMOSSE_create()
elif tracker_type == 'CSRT':
    tracker = cv2.TrackerCSRT_create()
else:
    print("Invalid tracker type!")
    sys.exit(1)

# Start video capture
video = cv2.VideoCapture("video/sample_2.mp4")  # Use 0 for webcam or provide video file path

# Check if the video was opened correctly
if not video.isOpened():
    print("Error: Could not open video.")
    sys.exit(1)

# Read the first frame of the video
ret, frame = video.read()
if not ret:
    print("Error: Failed to read the video frame.")
    sys.exit(1)

# Let the user select the ROI (Region of Interest) in the first frame
roi = cv2.selectROI("Select Object to Track", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Object to Track")  # Close the ROI window after selection

# Initialize the tracker with the selected ROI
tracker.init(frame, roi)

while True:
    # Read a new frame
    ret, frame = video.read()
    if not ret:
        break  # If the video has ended, break the loop

    # Update the tracker
    ret, roi = tracker.update(frame)

    # Draw the tracked object (bounding box) on the frame
    if ret:
        x, y, w, h = [int(v) for v in roi]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a green rectangle
        cv2.putText(frame, f"{tracker_type} Tracker", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        cv2.putText(frame, "Tracking failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with the tracked object
    cv2.imshow("Object Tracking", frame)

    # Exit condition on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close any OpenCV windows
video.release()
cv2.destroyAllWindows()
