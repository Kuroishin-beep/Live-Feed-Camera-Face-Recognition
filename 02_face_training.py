import sys
import time
import cv2
import numpy as np
import os
import schedule

# Path to the dataset
src_folder = r"D:\Github\Project\Live-Feed-Camera-Face-Recognition\01_Training_Dataset"
subfolders = ['Julia', 'Kiara', 'Sean', 'Tads']

# Initialize LBPH Recognizer & Haar Cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Progress bar function
def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    percent = "{0:.1f}".format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()

import cv2
import numpy as np
import os

def detect_faces_dnn(image, net):
    # Ensure image is in BGR format
    if len(image.shape) == 2:  
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0), swapRB=False)
    net.setInput(blob)
    detections = net.forward()
    
    h, w = image.shape[:2]
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            faces.append((x, y, x2 - x, y2 - y))
    
    return faces

def preprocess_image(image):
    if len(image.shape) == 3:  # If the image is colored (BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.equalizeHist(image)  # Normalize brightness
    image = cv2.GaussianBlur(image, (5, 5), 0)  # Reduce noise
    return image

def check_model_files(model_dir):
    if not os.path.exists(os.path.join(model_dir, "deploy.prototxt")):
        print("[ERROR] deploy.prototxt not found!")
    if not os.path.exists(os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")):
        print("[ERROR] res10_300x300_ssd_iter_140000.caffemodel not found!")

def detect_faces_haar(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Function to collect face data
def getImagesAndLabels(src_folder, subfolders):
    faceSamples, ids = [], []
    name_to_id = {name: idx + 1 for idx, name in enumerate(subfolders)}

    total_images = sum(
        len([f for f in os.listdir(os.path.join(src_folder, name)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for name in subfolders if os.path.exists(os.path.join(src_folder, name))
    )

    image_count = 0
    for name in subfolders:
        folder_path = os.path.join(src_folder, name)
        if not os.path.exists(folder_path):
            print(f"[WARNING] Folder {folder_path} does not exist, skipping...")
            continue

        imagePaths = sorted(
            [os.path.join(folder_path, f) for f in os.listdir(folder_path)
             if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.startswith(name)],
            key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0
        )

        for imagePath in imagePaths:
            try:
                image_count += 1
                print_progress_bar(image_count, total_images, prefix='Training Progress', suffix='Complete')

                # Load and preprocess Image
                img = cv2.imread(imagePath)  # Load the image properly
                if img is None:
                    print(f"[ERROR] Skipping {imagePath}: Corrupted or unreadable.")
                    continue

                img_numpy = preprocess_image(img)  # Now preprocess the loaded image
                face = detect_faces_haar(img)  # Detect faces using Haar Cascade (or DNN if you choose)

                if len(face) == 0:
                    print(f"[WARNING] No face detected in {imagePath}, skipping...")
                    continue

                for (x, y, w, h) in face:
                    faceSamples.append(img_numpy[y:y+h, x:x+w])  # Extract face region
                    ids.append(name_to_id[name])

            except Exception as e:
                print(f"[ERROR] Error processing {imagePath}: {e}")

    return faceSamples, ids


# Function to train the model
def train_model():
    print("\n[INFO] Training faces. This process runs every 30 minutes...")
    faces, ids = getImagesAndLabels(src_folder, subfolders)

    if len(faces) > 0:
        recognizer.train(faces, np.array(ids))

        # Ensure model directory exists
        model_path = r"D:\Github\Project\Live-Feed-Camera-Face-Recognition\Model"
        os.makedirs(model_path, exist_ok=True)

        recognizer.write(os.path.join(model_path, "face_recognizer.yml"))
        print(f"\n[INFO] {len(np.unique(ids))} faces trained. Model saved at {model_path}")
    else:
        print("\n[ERROR] No faces detected. Training aborted.")

# Initial Training
print("\n[INFO] Initial face training started. This may take a few seconds...")
faces, ids = getImagesAndLabels(src_folder, subfolders)

if len(faces) > 0:
    recognizer.train(faces, np.array(ids))
    
    model_path = r"D:\Github\Project\Live-Feed-Camera-Face-Recognition\Model"
    os.makedirs(model_path, exist_ok=True)

    recognizer.write(os.path.join(model_path, 'face_recognizer.yml'))
    print(f"\n[INFO] {len(np.unique(ids))} faces trained. Model saved at {model_path}")

    # Schedule the training every 30 minutes
    schedule.every(30).minutes.do(train_model)
    print("[INFO] Automated face training started. Training will run every 30 minutes.")

    train_model()  # Initial training on startup

    while True:
        schedule.run_pending()
        time.sleep(1)
else:
    print("\n[ERROR] No faces detected. Training aborted.")
