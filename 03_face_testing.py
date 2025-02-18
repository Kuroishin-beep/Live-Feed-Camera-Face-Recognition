import cv2
import numpy as np
import os

# Configuration
MODEL_PATH = r"D:\Github\Project\Live-Feed-Camera-Face-Recognition\Model\face_recognizer.yml"
DEFAULT_IMAGE_PATH = r"D:\Github\Project\Live-Feed-Camera-Face-Recognition\01_Training_Dataset\dataset\Sean\Sean.4.20.jpg"
CONFIDENCE_THRESHOLD = 10  # Minimum confidence percentage to accept a prediction

# Face ID-to-Name Mapping
FACE_MAP = {
    0: "Angelica",
    1: "Julia",
    2: "Kiara",
    3: "Sean"
}

def load_model(model_path):
    """Load the face recognition model."""
    if not os.path.exists(model_path):
        print(f"[ERROR] Face recognition model not found at {model_path}. Train the model first.")
        exit()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    print("[INFO] Face recognition model loaded successfully.")
    return recognizer

def initialize_face_detector():
    """Initialize the face detector."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        print("[ERROR] Haar cascade file not found.")
        exit()
    print("[INFO] Face detector initialized.")
    return cv2.CascadeClassifier(cascade_path)

def preprocess_face(face):
    """Preprocess a detected face for recognition."""
    face_resized = cv2.resize(face, (100, 100))  # Resize to the same size used during training
    face_normalized = cv2.normalize(face_resized, None, 0, 255, cv2.NORM_MINMAX)  # Normalize pixel values
    return face_normalized

def resize_image(img, width=800):
    """Resize an image while maintaining aspect ratio."""
    height = int((width / img.shape[1]) * img.shape[0])
    return cv2.resize(img, (width, height))

def recognize_face(input_image_path, recognizer, face_detector):
    """Recognize faces in the given image."""
    if not os.path.exists(input_image_path):
        print(f"[ERROR] Image not found at {input_image_path}")
        return

    img = cv2.imread(input_image_path)
    if img is None:
        print(f"[ERROR] Unable to read image: {input_image_path}")
        return

    # Preprocess the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Equalize histogram for better detection

    # Detect faces in the image
    faces_detected = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,  # Adjust to balance speed and accuracy
        minNeighbors=8,   # Higher value reduces false positives
        minSize=(40, 40)  # Minimum face size to detect
    )

    if len(faces_detected) == 0:
        print("[WARNING] No face detected in the image.")
    else:
        print(f"[INFO] {len(faces_detected)} face(s) detected.")
        for (x, y, w, h) in faces_detected:
            face = gray[y:y+h, x:x+w]
            face_preprocessed = preprocess_face(face)
            label, confidence = recognizer.predict(face_preprocessed)
            confidence_score = round(100 - confidence, 2)  # Convert to percentage

            # Log the confidence score for debugging
            print(f"[DEBUG] Face detected with confidence score: {confidence_score}% (Label: {label})")

            # Determine face label
            if confidence_score > CONFIDENCE_THRESHOLD:
                name = FACE_MAP.get(label, "Unknown")
            else:
                name = "Unknown"

            # Draw rectangle and label on the image
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, f"{name} ({confidence_score}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

            # Log the result
            if name == "Unknown":
                print(f"[WARNING] Low confidence: Detected face with confidence {confidence_score}%. Label is 'Unknown'.")
            else:
                print(f"[INFO] Recognized: {name} with confidence {confidence_score}%.")
    
    # Resize and display the image
    img_resized = resize_image(img)
    cv2.imshow("Recognized Faces", img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load model and face detector
    recognizer = load_model(MODEL_PATH)
    face_detector = initialize_face_detector()

    # Specify the input image path
    input_image_path = DEFAULT_IMAGE_PATH

    # Run face recognition
    recognize_face(input_image_path, recognizer, face_detector)
