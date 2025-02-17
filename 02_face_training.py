import os
import cv2
import numpy as np

# Define paths for the dataset and model
DATASET_DIR = r"D:\Github\Project\Live-Feed-Camera-Face-Recognition\01_Training_Dataset"
MODEL_DIR = r"D:\Github\Project\Live-Feed-Camera-Face-Recognition\Model"
MODEL_PATH = os.path.join(MODEL_DIR, "face_recognizer.yml")

# Create the model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Initialize the face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to prepare the training data
def prepare_training_data():
    print("[INFO] Preparing training data...")
    faces = []
    labels = []
    label_map = {}

    # Iterate over users (subfolders)
    for user_id, user_name in enumerate(os.listdir(DATASET_DIR)):
        user_folder = os.path.join(DATASET_DIR, user_name)
        
        # Only process folders
        if os.path.isdir(user_folder):
            label_map[user_id] = user_name
            print(f"[INFO] Processing images for {user_name} (ID: {user_id})")

            for file_name in os.listdir(user_folder):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Check if filename starts with the user's name, e.g., "Julia_1"
                    if file_name.startswith(user_name):
                        img_path = os.path.join(user_folder, file_name)
                        img = cv2.imread(img_path)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces_detected = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                        # If faces are detected, add them to training data
                        for (x, y, w, h) in faces_detected:
                            face = gray[y:y+h, x:x+w]
                            faces.append(face)
                            labels.append(user_id)

    return faces, labels, label_map

# Train the model
faces, labels, label_map = prepare_training_data()
recognizer.train(faces, np.array(labels))

# Save the trained model
recognizer.save(MODEL_PATH)
print(f"[INFO] Training complete. Model saved at {MODEL_PATH}")
