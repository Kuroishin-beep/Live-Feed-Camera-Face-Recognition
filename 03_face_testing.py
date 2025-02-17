import cv2
import numpy as np
import os

# Load the trained model
MODEL_PATH = r"D:\Github\Project\Live-Feed-Camera-Face-Recognition\Model\face_recognizer.yml"

# Ensure the model file exists before loading
if not os.path.exists(MODEL_PATH):
    print("[ERROR] Face recognition model not found! Train the model first.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

# Initialize the face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the label map (user IDs to names)
FACE_MAP = {
    1: "Julia",
    2: "Sean",
    3: "Kiara",
    4: "Angelica"
}

# Function to resize the image
def resize_image(img, width=800):
    height = int((width / img.shape[1]) * img.shape[0])
    return cv2.resize(img, (width, height))

# Function to recognize a face from an image
def recognize_face(input_image_path):
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"[ERROR] Unable to read image: {input_image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast for better detection

    # Improved face detection parameters
    faces_detected = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,   # Detect smaller faces
        minNeighbors=6,    # Reduce false positives
        minSize=(50, 50)   # Ignore very small faces
    )

    if len(faces_detected) == 0:
        print("[WARNING] No face detected in the image.")
    else:
        for (x, y, w, h) in faces_detected:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (100, 100))  # Standardize size
            label, confidence = recognizer.predict(face_resized)
            confidence_score = round(100 - confidence, 2)  # Convert to percentage

            # If confidence is low, classify as "Unknown"
            name = FACE_MAP.get(label, "Unknown") if confidence_score > 50 else "Unknown"

            # Draw rectangle and display name
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for recognized, red for unknown
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, f"{name} ({confidence_score}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

    # Resize and display the image
    img_resized = resize_image(img)
    cv2.imshow("Recognized Face", img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Run image recognition ---
if __name__ == "__main__":
    input_image_path = r"D:\Github\Project\Live-Feed-Camera-Face-Recognition\01_Training_Dataset\Julia\Julia1.jpg"
    recognize_face(input_image_path)
