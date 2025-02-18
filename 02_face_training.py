import sys
import time
import cv2
import numpy as np
import os
import schedule

# Path to the dataset
src_folder = r"D:\Github\Project\Live-Feed-Camera-Face-Recognition\01_Training_Dataset\dataset"
subfolders = ['Angelica', 'Julia', 'Kiara', 'Sean']

# Initialize LBPH Recognizer & Haar Cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Check if DNN model files are present
def check_model_files(model_dir):
    if not os.path.exists(os.path.join(model_dir, "deploy.prototxt")):
        print("[ERROR] deploy.prototxt not found!")
    if not os.path.exists(os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")):
        print("[ERROR] res10_300x300_ssd_iter_140000.caffemodel not found!")

# Load the DNN face detection model
def load_dnn_model(model_dir):
    deploy_file = os.path.join(model_dir, "deploy.prototxt")
    model_file = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    
    if not os.path.exists(deploy_file) or not os.path.exists(model_file):
        print("[ERROR] Missing DNN model files!")
        return None
    
    net = cv2.dnn.readNetFromCaffe(deploy_file, model_file)
    return net


# Progress bar function
def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()

# Image preprocessing
def preprocess_image(image, resize_width=800):
    if len(image.shape) == 3:  # If the image is colored (BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.equalizeHist(image)  # Normalize brightness
    image = cv2.GaussianBlur(image, (5, 5), 0)  # Reduce noise

    # Resize the image while maintaining aspect ratio
    height = int((resize_width / image.shape[1]) * image.shape[0])
    image = cv2.resize(image, (resize_width, height))

    # Normalize pixel values
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    return image


# DNN-based face detection
def detect_faces_dnn(image, net):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            faces.append((x, y, x2 - x, y2 - y))
    
    return faces

# Haar Cascade-based face detection
def detect_faces_haar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Function to collect face data
def getImagesAndLabels(src_folder, subfolders, net=None):
    faceSamples, ids = [], []
    name_to_id = {name: idx + 1 for idx, name in enumerate(subfolders)}
    total_images = sum(len(os.listdir(os.path.join(src_folder, name))) for name in subfolders if os.path.exists(os.path.join(src_folder, name)))
    image_count = 0

    for name in subfolders:
        folder_path = os.path.join(src_folder, name)
        for image_name in os.listdir(folder_path):
            imagePath = os.path.join(folder_path, image_name)
            img = cv2.imread(imagePath)

            if img is None:
                print(f"[WARNING] Could not read image {imagePath}, skipping...")
                continue

            img_numpy = preprocess_image(img)

            if net:
                faces = detect_faces_dnn(img, net)
            else:
                faces = detect_faces_haar(img)

            if len(faces) == 0:
                print(f"[WARNING] No face detected in {imagePath}, skipping...")
                continue

            for (x, y, w, h) in faces:
                face = img_numpy[y:y + h, x:x + w]
                if face.size > 0:
                    faceSamples.append(face)
                    ids.append(name_to_id[name])

            image_count += 1
            print_progress_bar(image_count, total_images, prefix='Processing images', suffix='Complete')

    return faceSamples, ids


def train_model(net=None):
    print("\n[INFO] Training faces. This process runs every 30 minutes...")
    faces, ids = getImagesAndLabels(src_folder, subfolders, net)

    total_faces = len(faces)
    if total_faces > 0:
        for i in range(total_faces):
            # Ensure that the face data is not empty and has valid dimensions
            if faces[i] is not None and faces[i].size > 0 and len(faces[i].shape) == 2:
                recognizer.update([faces[i]], np.array([ids[i]]))
                print_progress_bar(i + 1, total_faces, prefix='Training progress', suffix='Complete', length=50)
            else:
                print(f"[WARNING] Skipping invalid face data at index {i}")
        
        print("\n[INFO] Model training complete.")
        model_path = r"D:\Github\Project\Live-Feed-Camera-Face-Recognition\Model"
        recognizer.save(os.path.join(model_path, "face_recognizer.yml"))
        print(f"[INFO] Model saved at {model_path}")
    else:
        print("[ERROR] No faces found for training.")

# Initial Training
print("\n[INFO] Initial face training started. This may take a few seconds...")

# Check model files
model_dir = r"D:\Github\Project\Live-Feed-Camera-Face-Recognition\Model"
check_model_files(model_dir)

# Load DNN model
net = load_dnn_model(model_dir)

# Start initial face training
train_model(net)

# Schedule the training every 30 minutes
schedule.every(30).minutes.do(train_model, net=net)
print("[INFO] Automated face training started. Training will run every 30 minutes.")

# Start the scheduling loop
while True:
    schedule.run_pending()
    time.sleep(1)