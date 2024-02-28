import os

def has_subdirectories(directory_path):
    return any(os.path.isdir(os.path.join(directory_path, item)) for item in os.listdir(directory_path))
