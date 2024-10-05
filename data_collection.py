import cv2 as cv
import os
from mediapipe_utils import mediapipe_detection, draw_landmarks, extract_keypoints, mp_holistic, np

# Set up the data path
DATA_PATH = os.path.join('MediaPipe_Data')

# Function to dynamically create directories for new gestures
def setup_directories(new_action, no_of_sequences):
    for sequence in range(no_of_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, new_action, str(sequence)))
        except FileExistsError:
            pass

# Function to collect data for training (with 30 frames per sequence)
def collect_data(new_action, no_of_sequences=30):
    sequence_length = 30  # Fixed number of frames per sequence
    cap = cv.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        setup_directories(new_action, no_of_sequences)
        
        for sequence in range(no_of_sequences):
            # Countdown before each sequence collection
            for countdown in range(3, 0, -1):
                ret, frame = cap.read()
                cv.putText(frame, f'Starting in {countdown}', (200, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
                cv.imshow('Frame', frame)
                cv.waitKey(1000)  # 1 second delay

            for frame_num in range(sequence_length):  # Use fixed sequence length of 30
                ret, frame = cap.read()
                if not ret:
                    break
                # Perform mediapipe detection
                image, results = mediapipe_detection(frame, holistic)
                draw_landmarks(image, results)

                # Show collection notification
                if frame_num == 0:
                    cv.putText(image, f'STARTING COLLECTION for {new_action}, Sequence {sequence}', (10, 30), 
                               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
                    cv.waitKey(2000)  # Wait 2 seconds to get ready
                else:
                    cv.putText(image, f'Collecting frames for {new_action}, Sequence {sequence}', (10, 30), 
                               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

                # Extract and save keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, new_action, str(sequence), f'frame_{frame_num}.npy')
                np.save(npy_path, keypoints)

                cv.imshow('Frame', image)
                if cv.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv.destroyAllWindows()
