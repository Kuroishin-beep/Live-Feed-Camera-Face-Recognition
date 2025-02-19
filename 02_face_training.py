import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# Define paths and initialize models
dataset_path = r'D:\Github\Project\Live-Feed-Camera-Face-Recognition\01_Training_Dataset\dataset'
model_path = 'face_recognition_model.pkl'
label_encoder_path = 'label_encoder.pkl'
resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Label mapping
label_map = {'Angelica': 0, 'Julia': 1, 'Kiara': 2, 'Sean': 3}
X, y = [], []

def extract_single_face(image_path):
    """
    Attempts to detect a single face in an image. Adjusts detection thresholds
    dynamically to ensure only one face is detected. Skips the image if unable
    to meet the criterion after multiple attempts.

    Parameters:
    - image_path: str, path to the image file.

    Returns:
    - face: np.array, preprocessed face image suitable for embedding extraction.
    - None: if a single face could not be detected.
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    attempts = 0
    thresholds = [0.6, 0.7, 0.7]  # Initial confidence thresholds for PNet, RNet, ONet
    while attempts < 10:
        detector = MTCNN()
        detections = detector.detect_faces(img_rgb)
        face_count = len(detections)
        print(f"Attempt {attempts + 1}: Detected {face_count} face(s) in {image_path}")
        
        if face_count == 1:
            x, y, width, height = detections[0]['box']
            face = img_rgb[y:y+height, x:x+width]
            face = cv2.resize(face, (224, 224))
            face = preprocess_input(face.astype(np.float32))
            return face
        
        # Adjust confidence thresholds for the next attempt
        thresholds = [min(thresh + 0.05, 0.95) for thresh in thresholds]
        attempts += 1
    
    print(f"Skipping {image_path}: Unable to detect a single face after {attempts} attempts.")
    return None

# Process each person's images
for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if os.path.isdir(person_path) and person in label_map:
        for file in os.listdir(person_path):
            if file.lower().endswith(('.jpg', '.png')):
                image_path = os.path.join(person_path, file)
                face = extract_single_face(image_path)
                if face is not None:
                    embedding = resnet.predict(np.expand_dims(face, axis=0))
                    X.append(embedding.flatten())
                    y.append(label_map[person])

# Encode labels and train SVM
if X and y:  # Ensure there is data to train on
    X, y = np.array(X), np.array(y)
    le = LabelEncoder()
    y = le.fit_transform(y)
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X, y)

    # Save the trained model and label encoder
    dump(svm, model_path)
    dump(le, label_encoder_path)
    print("Model trained and saved successfully!")
else:
    print("No valid training data found. Model was not trained.")