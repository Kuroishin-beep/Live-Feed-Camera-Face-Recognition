import os
from pillow_heif import register_heif_opener
from PIL import Image

# Register HEIC support
register_heif_opener()

# Define paths
input_folder = "Uncleaned_Dataset"      # Folder where HEIC images are stored
output_folder = "Cleaned_Dataset" # Output folder for converted JPG images

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate through each person's folder inside dataset/
for person in os.listdir(input_folder):
    person_input_path = os.path.join(input_folder, person)
    person_output_path = os.path.join(output_folder, person)

    # Check if it's a directory (skip files)
    if os.path.isdir(person_input_path):
        os.makedirs(person_output_path, exist_ok=True)  # Create corresponding output folder

        # Get existing count of JPGs in the output folder (to continue numbering)
        existing_files = [f for f in os.listdir(person_output_path) if f.endswith(".jpg")]
        count = len(existing_files) + 1  # Start numbering from the next available index

        # Convert each HEIC file inside the person's folder
        for file in os.listdir(person_input_path):
            if file.lower().endswith(".heic"):
                heic_path = os.path.join(person_input_path, file)
                jpg_path = os.path.join(person_output_path, f"{person}{count}.jpg")

                # Open and convert the HEIC image to JPG
                img = Image.open(heic_path)
                img.convert("RGB").save(jpg_path, "JPEG")
                print(f"Converted: {file} → {jpg_path}")

                count += 1  # Increment for next image

print("HEIC to JPG conversion complete!")
