import os
import shutil

def copy_files(source_folder, destination_folder, start_num, end_num, new_prefix="Tads", new_start_num=66, file_ext=".jpg"):
    # Ensure destination directory exists
    os.makedirs(destination_folder, exist_ok=True)
    
    new_num = new_start_num  # Start renaming from new_start_num
    for i in range(start_num, end_num + 1):
        old_file_name = f"Angelica_{i}{file_ext}"
        source_file = os.path.join(source_folder, old_file_name)
        new_file_name = f"{new_prefix}{new_num}{file_ext}"
        destination_file = os.path.join(destination_folder, new_file_name)
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, destination_file)
            print(f"Copied and Renamed: {old_file_name} -> {new_file_name}")
            new_num += 1  # Increment new numbering
        else:
            print(f"File not found: {old_file_name}")

def main():
    source_folder = "Cleaned_Dataset/Angelica_New"
    destination_folder = "01_Training_Dataset/Angelica"
    start_num = 1
    end_num = 58
    new_start_num = 66  
    
    copy_files(source_folder, destination_folder, start_num, end_num, new_start_num=new_start_num)

if __name__ == "__main__":
    main()
