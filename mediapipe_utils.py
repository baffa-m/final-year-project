import cv2 as cv
import numpy as np
import mediapipe as mp

# Initialize MediaPipe holistic and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Function to perform MediaPipe detection
def mediapipe_detection(image, model):
    # Convert color space from BGR to RGB (MediaPipe uses RGB)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False  # Set flag to improve performance
    results = model.process(image)  # Make predictions using the holistic model
    image.flags.writeable = True  # Set flag back to True so we can manipulate the image again
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
    return image, results

# Function to draw landmarks on the image
def draw_landmarks(image, results):
    # Draw face landmarks
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
        mp_drawing_styles.get_default_face_mesh_tesselation_style())

    # Draw pose landmarks
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # Draw hand landmarks (left hand)
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Draw hand landmarks (right hand)
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# Function to extract keypoints from results
def extract_keypoints(results):
    # Extract pose landmarks (if available)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]
                    ).flatten() if results.pose_landmarks else np.zeros(33*4)

    # Extract face landmarks (if available)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
                    ).flatten() if results.face_landmarks else np.zeros(468*3)

    # Extract left-hand landmarks (if available)
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
                         ).flatten() if results.left_hand_landmarks else np.zeros(21*3)

    # Extract right-hand landmarks (if available)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
                          ).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    # Concatenate pose, face, left hand, and right hand landmarks into a single array
    return np.concatenate([pose, face, left_hand, right_hand])
