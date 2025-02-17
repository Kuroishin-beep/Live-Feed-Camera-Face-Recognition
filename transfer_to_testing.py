import os
import shutil

def copy_files(source_folder, destination_folder, start_num, end_num, prefix="Group (testing)", file_ext=".jpg"):
    # Ensure destination directory exists
    os.makedirs(destination_folder, exist_ok=True)
    
    for i in range(start_num, end_num + 1):
        file_name = f"{prefix}{i}{file_ext}"
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(destination_folder, file_name)
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, destination_file)
            print(f"Copied: {file_name}")
        else:
            print(f"File not found: {file_name}")

def main():
    source_folder = "Cleaned_Dataset/Group (testing)"
    destination_folder = "02_Testing_Dataset/Group (testing)"
    start_num = 1
    end_num = 18
    
    copy_files(source_folder, destination_folder, start_num, end_num)

if __name__ == "__main__":
    main()
