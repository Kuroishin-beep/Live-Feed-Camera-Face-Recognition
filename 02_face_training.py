import sys
import time
import cv2
import numpy as np
from PIL import Image
import os
import schedule

# Base path for face image dataset
src_folder = r"D:\Github\Project\Live-Feed-Camera-Face-Recognition\01_Training_Dataset"
subfolders = ['Julia', 'Kiara', 'Sean', 'Tads']

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Function to display progress percentage
def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()

# Function to get images and labels from subfolders
def getImagesAndLabels(src_folder, subfolders):
    faceSamples = []
    ids = []
    name_to_id = {name: idx + 1 for idx, name in enumerate(subfolders)}
    total_images = sum(
        len([f for f in os.listdir(os.path.join(src_folder, name)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for name in subfolders if os.path.exists(os.path.join(src_folder, name))
    )
    
    image_count = 0  # Track progress
    for name in subfolders:
        folder_path = os.path.join(src_folder, name)
        if not os.path.exists(folder_path):
            print(f"[WARNING] Folder {folder_path} does not exist, skipping...")
            continue
        # Find images that match "<name>1", "<name>2", etc.
        imagePaths = sorted(
            [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.startswith(name)],
            key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0
        )
        imagePaths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.startswith(name)]
        
        for imagePath in imagePaths:
            try:
                image_count += 1  # Increment progress
                print_progress_bar(image_count, total_images, prefix='Training Progress', suffix='Complete')

                # Load image and convert to grayscale
                PIL_img = Image.open(imagePath).convert('L')
                img_numpy = np.array(PIL_img, 'uint8')

                # Check if image is empty or unsupported format
                if img_numpy is None or img_numpy.size == 0:
                    print(f"[ERROR] Skipping {imagePath}: Empty or unsupported image format.")
                    continue

                id = name_to_id[name]  # Assign numeric ID based on folder name
                faces = detector.detectMultiScale(img_numpy, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) == 0:
                    print(f"[WARNING] No face detected in {imagePath}, skipping...")
                    continue
                
                for (x, y, w, h) in faces:
                    face = img_numpy[y:y + h, x:x + w]  # Extract face region
                    faceSamples.append(face)
                    ids.append(id)
            except Exception as e:
                print(f"[ERROR] Error processing {imagePath}: {e}")
    return faceSamples, ids

#Function to train the model
def train_model():
    print("\n[INFO] Training faces. This process runs every 30 minutes...")
    faces, ids = getImagesAndLabels(src_folder, subfolders)
        
    if len (faces) > 0:
        recognizer.train(faces, np.array(ids))
        recognizer.write(os.path.join(model_path, "face_recognizer.yml"))
        print(f"\n[INFO] {len(np.unique(ids))} faces trained. Model saved at {model_path}")
    else:
        print("\n[ERROR] No faces detected. Training aborted.")
            
            
print("\n[INFO] Training faces. It will take a few seconds. Wait...")
faces, ids = getImagesAndLabels(src_folder, subfolders)

if len(faces) > 0:
    recognizer.train(faces, np.array(ids))
    
    # Ensure model directory exists
    model_path = r"D:\Github\Project\Live-Feed-Camera-Face-Recognition\Model"
    os.makedirs(model_path, exist_ok=True)
    
    # Save the trained model
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
