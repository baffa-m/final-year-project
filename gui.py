import tkinter as tk
from tkinter import simpledialog
from data_collection import collect_data
from model import train_model  # Import the train_model function from model.py
from mouse_control import control_mouse
from model import detect_gestures

# Function to start the Tkinter GUI
def start_gui():
    root = tk.Tk()
    root.title("Gesture Recognition Trainer")
    root.geometry("300x300")

    def start_training():
        new_action = simpledialog.askstring("Input", "Enter the new gesture name:", parent=root)
        no_of_sequences = simpledialog.askinteger("Input", "Enter the number of sequences:", parent=root)

        if new_action and no_of_sequences:
            print(f"Training for '{new_action}' with {no_of_sequences} sequences of 30 frames each.")
            collect_data(new_action, no_of_sequences)

    def retrain_model():
        print("Training model based on all collected data...")
        train_model()  # This will retrain the model with all collected data and save a new action.h5

    # Create buttons for each mode
    train_button = tk.Button(root, text="Collect Gesture Data", command=start_training)
    detect_button = tk.Button(root, text="Detection Mode", command=detect_gestures)
    mouse_button = tk.Button(root, text="Mouse Control Mode", command=control_mouse)
    retrain_button = tk.Button(root, text="Train Model", command=retrain_model)  # Button to retrain the model

    # Add buttons to the GUI
    train_button.pack(pady=20)
    detect_button.pack(pady=20)
    mouse_button.pack(pady=20)
    retrain_button.pack(pady=20)  # Add retrain button to GUI

    root.mainloop()
