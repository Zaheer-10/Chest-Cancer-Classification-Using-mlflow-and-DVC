import shutil
import os

def copy_file_to_destination(source_file, destination_folder):
    try:
        # Get the absolute paths by joining with the current working directory
        cwd = os.getcwd()  # Get current working directory
        source_file_path = os.path.join(cwd, source_file)
        destination_folder_path = os.path.join(cwd, destination_folder)

        # Create the destination directory if it doesn't exist
        if not os.path.exists(destination_folder_path):
            os.makedirs(destination_folder_path)

        destination_file = os.path.join(destination_folder_path, os.path.basename(source_file))

        shutil.copy(source_file_path, destination_file)
        print(f"File '{os.path.basename(source_file)}' copied successfully to '{destination_folder_path}'")
    except FileNotFoundError:
        print("File not found or directories are incorrect.")
    except PermissionError:
        print("Permission denied to copy the file.")
    except shutil.SameFileError:
        print("Source and destination represent the same file.")
    except Exception as e:
        print(f"An error occurred: {e}")

source_file_path = "artifacts/training/model.h5"
destination_folder_path = "model"
copy_file_to_destination(source_file_path, destination_folder_path)
