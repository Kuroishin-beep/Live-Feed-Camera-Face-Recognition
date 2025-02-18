"""
Organize and assign IDs to faces from the dataset for training
==> The script will rename and structure the dataset for training
==> Each image will have a unique ID based on the user (Julia, Sean, Kiara, Angelica)

Modified for pre-existing dataset use.
"""

import os
import shutil

# Define the dataset paths
SOURCE_DIR = r"D:\Github\Project\Live-Feed-Camera-Face-Recognition\01_Training_Dataset"
TARGET_DIR = os.path.join(SOURCE_DIR, "dataset")  # Organized dataset folder

# List of users and their IDs
USERS = {
    "Angelica": 1,
    "Julia": 2,
    "Kiara": 3,
    "Sean": 4
}

# Create target directory if it doesn't exist
os.makedirs(TARGET_DIR, exist_ok=True)

# Function to process and rename images
def organize_dataset():
    print("[INFO] Organizing dataset...")
    for user_name, user_id in USERS.items():
        user_source_path = os.path.join(SOURCE_DIR, user_name)
        user_target_path = os.path.join(TARGET_DIR, user_name)
        
        if not os.path.exists(user_source_path):
            print(f"[WARNING] Source folder for {user_name} does not exist. Skipping...")
            continue
        
        # Create a target directory for the user
        os.makedirs(user_target_path, exist_ok=True)
        
        # Process images in the user's source directory
        count = 0
        for file_name in os.listdir(user_source_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
                count += 1
                source_file = os.path.join(user_source_path, file_name)
                target_file = os.path.join(user_target_path, f"{user_name}.{user_id}.{count}.jpg")
                
                # Copy and rename the file
                shutil.copy(source_file, target_file)
        
        print(f"[INFO] Processed {count} images for {user_name}.")
    
    print("[INFO] Dataset organization complete.")

# Run the script
if __name__ == "__main__":
    organize_dataset()
