import os
import io
import cv2
import requests
import numpy as np
import torch
from dotenv import load_dotenv
from roboflow import Roboflow
from urllib.parse import quote
from PIL import Image, ImageDraw

# Visualising prediction
def test_draw_bounding_boxes(image_path, predictions, output_directory):
    img = Image.open(image_path)
    drawn_image = img.copy()
    
    # Draw bounding boxes on the image copy
    draw = ImageDraw.Draw(drawn_image)
    for prediction in predictions['predictions']:
        x, y, width, height = (
            prediction['x'],
            prediction['y'],
            prediction['width'],
            prediction['height'],
        )
        confidence = prediction['confidence']

        # Can use confidence to filter out low-confidence detections if needed
        half_width = width / 2
        half_height = height / 2
        if confidence >= 0.7:  # Adjustable the threshold
            draw.rectangle([
                x - half_width, 
                y - half_height, 
                x + half_width, 
                y + half_height
                ], outline="red", width=2)
    
    # print number of bounding boxes
    print("Num boxes:\n", len(predictions['predictions']))

    # print bounding boxes
    # print("\nBounding boxes:", predictions['predictions'])

    # save output 
    output_image_path = os.path.join(output_directory, f"detection_{os.path.basename(image_path)}")
    drawn_image.save(output_image_path)

# Currently in development
# Yolov5 potential fixes:
# - increase sample size
# - increase epochs
# - pretrained?
def detect_yolov5(image_path, weights='yolo/detection/yolov5_weights/yolov5s.pt', confidence_threshold=0.5):
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5:v6.0', 'custom', path=weights)

    # Load image
    img = Image.open(image_path)
    original_image = img.copy()

    # Perform inference
    results = model(img, size=640)  # Change size as needed

    # Convert list of tensors to list of NumPy arrays
    text_results = [tensor.detach().cpu().numpy() for tensor in results.xyxy]

    # Concatenate NumPy arrays along axis 0
    text_results = np.concatenate(text_results, axis=0)

    # Filter results for text detection class 
    text_results = text_results[text_results[:, -1] == 0]  

    # Filter results by confidence threshold
    text_results = text_results[text_results[:, -2] > confidence_threshold]

    # Extract bounding boxes
    bounding_boxes = text_results[:, :-2]

    return bounding_boxes

# Roboflow's API usage - bugged
# Model Type: Roboflow 3.0 Object Detection (Fast)
# Checkpoint: COCOn
# Suitable for scanned manga
def detect_roboflow_manga_api(image_path):
    load_dotenv()
    ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY', '')

    # # Load and encode Image
    # with open(image_path, "rb") as image_file:
    #     base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    # url_encoded_image = quote(base64_image)

    # Load Image
    img = cv2.imread(image_path)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(image)

    # Convert to a high quality JPEG PIL image  
    buffered = io.BytesIO()
    pilImage.save(buffered, quality=100, format="JPEG")

    # Build multipart form and post request
    m = MultipartEncoder(fields={'file':("imageToUpload", buffered.getvalue(), "image/jpeg")})

    api_url = "".join([
        "https://detect.roboflow.com/image_tt/1",    
    ])
    payload = {
        'image': 1,
        'api_key': ROBOFLOW_API_KEY,
    }
    headers = {
        "Authorization": f"Bearer {ROBOFLOW_API_KEY}",
        'Content-Type': "application/json"
    }
    
    response = requests.post(api_url, data=payload, headers=headers)
    print (ROBOFLOW_API_KEY)

    return response.status_code

# Roboflow import - working
# Model Type: Roboflow 3.0 Object Detection (Fast)
# Checkpoint: COCOn
# Suitable for scanned manga
def detect_roboflow_manga(image_path):
    load_dotenv()
    ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY', '')

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace().project("image_tt")
    model = project.version('1').model

    # returns dictionary   
    return model.predict(image_path, confidence=50, overlap=50).json()
    
