import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from joblib import load
import os

# Load models
model_path = 'face_recognition_model.pkl'
detector = MTCNN()
resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
svm = load(model_path)

# Create a dictionary to map IDs to names
id_to_name = {
    0: "Angelica",
    1: "Julia",
    2: "Kiara",
    3: "Sean"
}

def identify_faces(image_path):
    if not os.path.exists(image_path):
        print(f"Error: The specified image path '{image_path}' does not exist.")
        return
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(img_rgb)

    if not detections:
        print("[INFO] No faces detected in the image.")
        return

    for det in detections:
        x, y, width, height = det['box']
        confidence = det['confidence']  # Get the confidence score of the detection
        confidence_percentage = confidence * 100  # Convert to percentage
        
        # Ensure coordinates are within image bounds
        x, y = max(0, x), max(0, y)
        width, height = max(0, width), max(0, height)

        face = img_rgb[y:y+height, x:x+width]
        face = cv2.resize(face, (224, 224))
        face = preprocess_input(face.astype(np.float32))
        embedding = resnet.predict(np.expand_dims(face, axis=0)).flatten()

        # Predict the label ID
        prediction = svm.predict([embedding])[0]  # Get the ID (0, 1, 2, etc.)
        label_name = id_to_name.get(prediction, "Unknown")  # Use the dictionary to map ID to name
        print(f"[INFO] Detected Name: {label_name}, Confidence: {confidence_percentage:.2f}%")  # Debugging

        # Draw bounding box and label on the image
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 8)
        text = f"{label_name} ({confidence_percentage:.2f}%)"
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)

    # Resize the image for display
    target_width = 800  # Set your desired width
    aspect_ratio = target_width / img.shape[1]
    resized_img = cv2.resize(img, (target_width, int(img.shape[0] * aspect_ratio)))

    cv2.imshow("Identified Faces", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test with a group image
image_path = r'D:\Github\Project\Live-Feed-Camera-Face-Recognition\02_Testing_Dataset\Group (testing)\Group (testing)3.jpg'  # Update with the correct path
identify_faces(image_path)
