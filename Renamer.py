
import os

def rename_files(directory, name):
    try:
        files = sorted(os.listdir(directory))  # Get and sort files
        for index, filename in enumerate(files, start=1):
            old_path = os.path.join(directory, filename)
            ext = os.path.splitext(filename)[1]  # Get file extension
            new_name = f"{name}{index}{ext}"
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)
        print("Renaming completed successfully!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    folder_path = input("Enter the folder path: ")
    base_name = input("Enter the base name: ")
    rename_files(folder_path, base_name)
