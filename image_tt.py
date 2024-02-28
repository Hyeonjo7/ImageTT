# Image Translator / Typesetter
import os
from PIL import Image
from utils import os_search
from detection import text_detection    

def collate_bounding_boxes(image_paths, output_directory):
    processed_bounding_boxes = []
    for image_path in image_paths:
        # Add bounding box info to the list
        bounding_boxes = text_detection.detect_roboflow_manga(image_path)
        processed_bounding_boxes.append(bounding_boxes)

        # TESTING through visualisation 
        text_detection.test_draw_bounding_boxes(image_path, bounding_boxes, output_directory)
    return processed_bounding_boxes

def process_images_in_folder(input_directory, output_directory, folder_name):
    folder_path = os.path.join(input_directory, folder_name)
    output_location = os.path.join(output_directory, folder_name)
    # check if folder exists in input
    if not os.path.isdir(folder_path):
        return 0
    
    # create matching folder if not in output
    if not os.path.exists(output_location):
        os.makedirs(output_location)

    # collect and sort image paths in the folder
    image_paths = sorted([
        os.path.join(folder_path, image) 
        for image in os.listdir(folder_path) 
        if image.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ])

    # Detect texts' bounding boxes
    # currently custom fit for roboflow
    # list of dictionaries containing more than just bounding boxes
    processed_bounding_boxes= collate_bounding_boxes(image_paths, output_directory)

    # for now it will return the processed bounding boxes
    return len(processed_bounding_boxes)

def search_input_directory(input_directory, output_directory):
    image_count_total = 0

    # search folders in the input directory
    if os_search.has_subdirectories(input_directory):
        for folder_name in os.listdir(input_directory):
            image_count_total += process_images_in_folder(input_directory, output_directory, folder_name)
    else:
        process_images_in_folder(input_directory, output_directory, "")
    
    print(f"Total images translated: {image_count_total}")

if __name__ == '__main__':
    input_directory = "/home/hyeonjo/personal/image-tt/test"
    output_directory = "/home/hyeonjo/personal/image-tt/test-output"

    search_input_directory(input_directory, output_directory)