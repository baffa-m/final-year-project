import cv2 as cv
import pyautogui
from mediapipe_utils import mediapipe_detection, draw_landmarks, mp_holistic

# Function to move the mouse using hand landmarks (e.g., index finger tip)
def move_mouse(results, invert_x=False, invert_y=False):
    screen_width, screen_height = pyautogui.size()
    if results.right_hand_landmarks:
        # Get the index finger tip coordinates
        index_finger_tip = results.right_hand_landmarks.landmark[8]
        
        # Invert mouse movements if needed
        x = (1 - index_finger_tip.x if invert_x else index_finger_tip.x) * screen_width
        y = (1 - index_finger_tip.y if invert_y else index_finger_tip.y) * screen_height

        pyautogui.moveTo(x, y)

# Function to detect gestures and perform mouse actions
def perform_gesture_actions(results):
    if results.right_hand_landmarks:
        # Detect pinch for dragging (thumb tip and index finger tip proximity)
        thumb_tip = results.right_hand_landmarks.landmark[4]
        index_finger_tip = results.right_hand_landmarks.landmark[8]

        # Calculate distance between thumb and index finger tip for drag detection
        distance = ((thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2) ** 0.5

        if distance < 0.05:
            pyautogui.mouseDown()  # Start dragging
        else:
            pyautogui.mouseUp()    # Release drag

        # Right-click (detect pinky and thumb together, for example)
        pinky_tip = results.right_hand_landmarks.landmark[20]
        if ((thumb_tip.x - pinky_tip.x) ** 2 + (thumb_tip.y - pinky_tip.y) ** 2) ** 0.5 < 0.05:
            pyautogui.click(button='right')

        # Double-click (use a custom gesture, like index and middle finger together)
        middle_finger_tip = results.right_hand_landmarks.landmark[12]
        if ((index_finger_tip.x - middle_finger_tip.x) ** 2 + (index_finger_tip.y - middle_finger_tip.y) ** 2) ** 0.5 < 0.05:
            pyautogui.doubleClick()

# Function to control the mouse
def control_mouse(invert_x=False, invert_y=False):
    cap = cv.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform detection with MediaPipe
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)

            # Control mouse movement
            move_mouse(results, invert_x, invert_y)

            # Perform specific gesture actions (e.g., right-click, drag, etc.)
            perform_gesture_actions(results)

            # Display the frame
            cv.putText(image, "Mouse Control Mode", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            cv.imshow('Mouse Control', image)

            if cv.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()

# Example usage:
# control_mouse(invert_x=True, invert_y=False)  # Inverts horizontal mouse movement
