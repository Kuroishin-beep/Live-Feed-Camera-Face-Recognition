import cv2
import numpy as np
from PIL import Image
import os

# Base path for face image dataset
src_folder = r"D:\Github\Project\Live-Feed-Camera-Face-Recognition\01_Training_Dataset"
subfolders = ['Julia', 'Kiara', 'Sean', 'Tads']

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Function to get images and labels from subfolders
def getImagesAndLabels(src_folder, subfolders):
    faceSamples = []
    ids = []
    name_to_id = {name: idx + 1 for idx, name in enumerate(subfolders)}
    
    for name in subfolders:
        folder_path = os.path.join(src_folder, name)
        if not os.path.exists(folder_path):
            print(f"[WARNING] Folder {folder_path} does not exist, skipping...")
            continue
        
        imagePaths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.startswith(name)]
        
        for imagePath in imagePaths:
            try:
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

                print(f"[INFO] Processing {imagePath} - Found {len(faces)} face(s)")
                
                for (x, y, w, h) in faces:
                    face = img_numpy[y:y + h, x:x + w]  # Extract face region
                    faceSamples.append(face)
                    ids.append(id)

                    # Display each detected face for verification
                    cv2.imshow(f"Training - {name}", face)
                    cv2.waitKey(100)  # Pause briefly to visualize
                
            except Exception as e:
                print(f"[ERROR] Error processing {imagePath}: {e}")

    cv2.destroyAllWindows()  # Close any open OpenCV windows
    return faceSamples, ids

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
else:
    print("\n[ERROR] No faces detected. Training aborted.")
