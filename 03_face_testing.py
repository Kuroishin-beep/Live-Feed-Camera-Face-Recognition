import cv2
import numpy as np

# Load the trained model
MODEL_PATH = r"D:\Github\Project\Live-Feed-Camera-Face-Recognition\Model\face_recognizer.yml"
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

# Initialize the face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the label map (user IDs to names)
LABEL_MAP = {
    1: "Julia",
    2: "Sean",
    3: "Kiara",
    4: "Angelica"
}

# Function to resize the image to fit the screen
def resize_image(img, width=800):
    height = int((width / img.shape[1]) * img.shape[0])
    return cv2.resize(img, (width, height))

# Function to recognize a face from an input image
def recognize_face(input_image_path):
    img = cv2.imread(input_image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces_detected:
        face = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face)
        name = LABEL_MAP.get(label, "Unknown")

        # Display the recognized face with its label
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"{name} ({round(confidence, 2)}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Resize the image to fit the screen/window
    img_resized = resize_image(img)

    # Display the result
    cv2.imshow("Recognized Face", img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage: Pass an image path for recognition
input_image_path = r"D:\Github\Project\Live-Feed-Camera-Face-Recognition\02_Testing_Dataset\Group (testing)\Group (testing)14.jpg"
recognize_face(input_image_path)
