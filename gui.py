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
    root.geometry("600x150") 

    title_label = tk.Label(root, text="Hand Gesture Recognition System", font=("Helvetica", 16))
    title_label.pack(pady=10)

    def start_training():
        new_action = simpledialog.askstring("Input", "Enter the new gesture name:", parent=root)
        no_of_sequences = simpledialog.askinteger("Input", "Enter the number of sequences:", parent=root)

        if new_action and no_of_sequences:
            print(f"Training for '{new_action}' with {no_of_sequences} sequences of 30 frames each.")
            collect_data(new_action, no_of_sequences)

    def retrain_model():
        print("Training model based on all collected data...")
        train_model()  # This will retrain the model with all collected data and save a new action.h5

    # Create a frame to hold the buttons in a single row
    button_frame = tk.Frame(root)
    button_frame.pack(pady=20)  # Add padding around the frame

    # Create buttons for each mode and add them to the frame
    train_button = tk.Button(button_frame, text="Collect Gesture Data", command=start_training)
    detect_button = tk.Button(button_frame, text="Detection Mode", command=detect_gestures)
    mouse_button = tk.Button(button_frame, text="Mouse Control Mode", command=lambda: control_mouse(invert_x=True, invert_y=False))
    retrain_button = tk.Button(button_frame, text="Train Model", command=retrain_model)  # Button to retrain the model

    # Pack buttons in a row inside the frame
    train_button.pack(side=tk.LEFT, padx=10)
    detect_button.pack(side=tk.LEFT, padx=10)
    mouse_button.pack(side=tk.LEFT, padx=10)
    retrain_button.pack(side=tk.LEFT, padx=10)

    root.mainloop()

# Run the GUI
start_gui()
