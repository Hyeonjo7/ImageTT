# Detection with EAST (Efficient and Accurate Scene Text) text detection model with OpenCV

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

def text_detection(image_path):
    # Load the image
    image = cv2.imread(image_path)
    orig = image.copy()

    # Set the path to the pre-trained EAST text detection model
    east_model_path = "path/to/east_model.pb"

    # Load the EAST model
    net = cv2.dnn.readNet(east_model_path)

    # Resize the image to meet the input size requirements of the EAST model
    (H, W) = image.shape[:2]
    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    # Resize the image and get the new dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # Define the two output layer names for the EAST model
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    # Construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)

    # Forward pass of the blob through the network
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # Decode the predictions and apply non-maxima suppression
    rectangles, confidences = decode_predictions(scores, geometry)
    indices = cv2.dnn.NMSBoxesRotated(rectangles, confidences, 0.5, 0.4)

    # Extract the text regions
    text_regions = [cv2.boxPoints(rectangles[i]) for i in indices]

    return text_regions

def decode_predictions(scores, geometry):
    # Number of rows and columns in the scores and geometry arrays
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # Loop over the number of rows
    for y in range(0, numRows):
        # Extract the scores and geometrical data
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # Loop over the number of columns
        for x in range(0, numCols):
            # Ignore low probability regions
            if scoresData[x] < 0.5:
                continue

            # Compute the offset factor as the current feature map pixel's (x, y)-coordinates
            offsetX = x * 4.0
            offsetY = y * 4.0

            # Extract the rotation angle for the prediction and compute its sine and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # Compute the box dimensions
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # Compute both the starting and ending (x, y)-coordinates for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # Add the bounding box coordinates and probability to the list
            rects.append(((startX, startY, endX, endY), scoresData[x]))
            confidences.append(scoresData[x])

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = cv2.dnn.NMSBoxesRotated(rects, confidences, 0.5, 0.4)
    result_rects = [rects[i][0] for i in indices]

    return result_rects, confidences

