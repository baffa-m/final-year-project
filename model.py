import os
import numpy as np
import cv2 as cv
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from mediapipe_utils import mediapipe_detection, draw_landmarks, extract_keypoints

# Path where the gesture data is stored
DATA_PATH = os.path.join('MediaPipe_Data')

# Define the model structure
def create_model(actions):
    sequence_length = 30  # Fixed sequence length of 30
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 1662)))  # Sequence length fixed
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(actions), activation='softmax'))  # Number of output actions
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

# Function to load existing gesture data
def load_existing_data():
    sequences, labels = [], []
    actions = []  # List of all actions/gestures
    sequence_length = 30  # Fixed number of frames per sequence

    # Check if existing gestures are present in DATA_PATH
    if os.path.exists(DATA_PATH):
        for action in os.listdir(DATA_PATH):
            actions.append(action)
            for sequence in os.listdir(os.path.join(DATA_PATH, action)):
                window = []
                for frame_num in range(sequence_length):  # Use fixed sequence length of 30
                    # Construct file path
                    file_path = os.path.join(DATA_PATH, action, sequence, f'frame_{frame_num}.npy')
                    
                    # Check if the file exists
                    if os.path.exists(file_path):
                        res = np.load(file_path)
                        window.append(res)
                    else:
                        print(f"Warning: {file_path} does not exist. Skipping this frame.")

                # Check if window is not empty and has the expected length
                if len(window) == sequence_length:  # Ensure it matches the fixed sequence length
                    sequences.append(window)
                    labels.append(actions.index(action))
                else:
                    print(f"Warning: Sequence for action '{action}' is incomplete.")

    return np.array(sequences, dtype=object), np.array(labels), actions

def prob_viz(res, actions, input_frame):
    """Visualize the prediction probabilities using a single color (blue)."""
    output_frame = input_frame.copy()
    color = (255, 0, 0)  # Blue in BGR format

    # Loop through each action and its probability, and visualize it on the frame
    for num, prob in enumerate(res):
        cv.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), color, -1)
        cv.putText(output_frame, f'{actions[num]}: {prob * 100:.2f}%', (0, 85 + num * 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        
    return output_frame

# Function to train the model on all collected data
def train_model():
    # Load existing gesture data
    X, y, actions = load_existing_data()

    if X.size == 0:
        print("No data available for training.")
        return

    # Ensure X and y are NumPy arrays with the correct types
    X = np.array(X).astype('float32')  # Convert to float32 for TensorFlow
    y = np.array(y).astype('int32')    # Convert to int32 for labels

    # One-hot encode labels
    y = to_categorical(y).astype(int)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create and train the model
    model = create_model(actions)
    model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))

    # Save the trained model weights as 'action.h5'
    model.save_weights('action.weights.h5')
    print("Model weights saved successfully to 'action.weights.h5'.")
    

# Function to detect gestures using the trained model
def detect_gestures():
    _, _, actions = load_existing_data()

    if len(actions) == 0:
        print("No actions available for detection.")
        return

    # Create the model with the dynamically generated actions
    model = create_model(actions)
    model.load_weights('action.weights.h5')  # Load the pre-trained weights

    cap = cv.VideoCapture(0)
    sequence = []  # Sequence to store the keypoints
    predictions = []  # To store recent predictions
    threshold = 50  # Confidence threshold
    sliding_window_size = 5  # Size of the sliding window for prediction consistency
    predicted_action = "No Gesture Detected"

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform detection with mediapipe
            image, results = mediapipe_detection(frame, holistic)

            # If no landmarks detected, reset predictions
            if not results.pose_landmarks and not results.face_landmarks and not results.left_hand_landmarks and not results.right_hand_landmarks:
                cv.putText(image, "No Gesture Detected", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
            else:
                draw_landmarks(image, results)

                # Extract keypoints and append to sequence
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]  # Keep only the last 30 frames

                # Once we have enough frames, make predictions
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    confidence = np.max(res) * 100

                    # If the confidence of the predicted gesture is above the threshold
                    if confidence > threshold:
                        predicted_action = actions[np.argmax(res)]
                        predictions.append(predicted_action)
                    else:
                        predictions.append("No Gesture Detected")

                    # Keep only the last N predictions
                    if len(predictions) > sliding_window_size:
                        predictions = predictions[-sliding_window_size:]

                    # Get the most frequent prediction in the sliding window
                    most_common_prediction = max(set(predictions), key=predictions.count)

                    # Only update the predicted action if it appears consistently in the window
                    if predictions.count(most_common_prediction) > sliding_window_size // 2:
                        predicted_action = most_common_prediction
                    else:
                        predicted_action = "No Gesture Detected"

                    # Display the prediction and confidence
                    cv.putText(image, f'Gesture: {predicted_action}', (10, 50),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
                    cv.putText(image, f'Confidence: {confidence:.2f}%', (10, 100),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

            # Show the frame
            cv.imshow('Gesture Recognition', image)

            if cv.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()
