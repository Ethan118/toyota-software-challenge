import cv2
import os
import time

# Create a folder to store the captured frames if it doesn't exist
output_folder = "captured_frames"
os.makedirs(output_folder, exist_ok=True)

# Open the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, change to 1, 2, etc. for other cameras

if not cap.isOpened():
    print("Error: Couldn't open the webcam.")
else:
    start_time = time.time()
    interval = 3  # Capture and save an image every 10 seconds

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Couldn't read a frame.")
            break

        # Resize the frame to 640x480
        frame = cv2.resize(frame, (640, 480))

        current_time = time.time()

        if current_time - start_time >= interval:
            # Save the frame to the output folder
            cv2.imwrite(os.path.join(output_folder, f"frame_{int(current_time)}.png"), frame)
            start_time = current_time

        # Display the frame
        cv2.imshow('Captured Frame', frame)

        # Press 'q' to quit and save the current frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
