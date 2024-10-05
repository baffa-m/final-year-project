import cv2 as cv
import pyautogui
from mediapipe_utils import mediapipe_detection, draw_landmarks, mp_holistic

# Function to move the mouse using hand landmarks (e.g., index finger tip)
def move_mouse(results):
    screen_width, screen_height = pyautogui.size()
    if results.right_hand_landmarks:
        # Get the index finger tip coordinates
        index_finger_tip = results.right_hand_landmarks.landmark[8]
        x = index_finger_tip.x * screen_width
        y = index_finger_tip.y * screen_height
        pyautogui.moveTo(x, y)

# Function to control the mouse
def control_mouse():
    cap = cv.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform detection with mediapipe
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)

            # Control mouse using hand gestures
            move_mouse(results)

            # Display the frame
            cv.putText(image, "Mouse Control Mode", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            cv.imshow('Mouse Control', image)

            if cv.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()
