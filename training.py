import cv2
import numpy as np
import os
import re

def load_images_from_folder(dataset_path):
    all_faces = []
    all_labels = []
    label2name = {"Angelica": 0, "Julia": 1, "Kiara": 2, "Sean": 3}
    name2label = {v: k for k, v in label2name.items()}
    
    total_images = 0
    for name, label in label2name.items():
        person_folder = os.path.join(dataset_path, name)
        if not os.path.isdir(person_folder):
            continue
        
        images_count = 0
        for file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                all_faces.append(img)
                all_labels.append(label)
                images_count += 1
                total_images += 1
        
        print(f"Loaded {images_count} images for {name} (Label {label})")
    
    print(f"\nTotal images loaded: {total_images}")
    return all_faces, all_labels, name2label

def train_face_recognizer():
    dataset_path = r"D:\Github\Project\Live-Feed-Camera-Face-Recognition\01_Training_Dataset\01_Dataset"
    print("Loading images from dataset...")
    all_faces, all_labels, name2label = load_images_from_folder(dataset_path)
    
    if len(all_faces) == 0:
        print("No face images found in dataset folder. Exiting training.")
        exit(1)
    
    print(f"Training the face recognizer with {len(all_faces)} images...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(all_faces, np.array(all_labels))
    recognizer.save("trainingData.yml")
    
    print("\nTraining complete. Model saved to 'trainingData.yml'.")
    print("Label mapping:", name2label)
    return name2label


def main():
    name2label = train_face_recognizer()
    print("Face recognition training finished. You can now test with new images.")

if __name__ == '__main__':
    main()
