from ultralytics import YOLO
import cv2
import numpy as np
import time
from scipy.io import loadmat


def calculateDetectionResults(boxes, central_points):
    detection_results = {}
    for point in central_points:
        point_x = point[0]
        point_y = point[1]
        detected = 0
        for box in boxes:
            xA = box[0]
            yA = box[1]
            xB = box[2]
            yB = box[3]
            if (point_x >= xA and point_x <= xB) and (point_y >= yA and point_y <= yA + (yB - yA) / 4):
                detected = 1
                break
            elif (point_x >= xA and point_x <= xB) and (point_y >= yA and point_y <= yA + (yB - yA)) and point_y >= 625:
                detected += 1
                break
        detection_results[str((point_x, point_y))] = detected
    return detection_results

def getHumanNumber(gt_path):
    # Load the .mat file
    mat = loadmat(gt_path)
    # Get the points
    human_number = mat['image_info'][0][0][0][0][1]
    # Convert from ndarray
    human_number = human_number.item()
    return human_number

def getHumanLocations(gt_path):
    # Load the .mat file
    mat = loadmat(gt_path)
    # Get the points
    central_points = mat['image_info'][0][0][0][0][0]
    return central_points

def drawCentralPoints(image, central_points):
    # Draw the points on the image
    for point in central_points:
        x, y = point[:2]
        cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)

def calculateAccuracy(boxes, central_points):
    correct_detections = 0
    for point in central_points:
        point_x = point[0]
        point_y = point[1]
        for box in boxes:
            xA = box[0]
            yA = box[1]
            xB = box[2]
            yB = box[3]
            if (point_x >= xA and point_x <= xB) and (point_y >= yA and point_y <= yA + (yB - yA) / 4):
                correct_detections += 1
                break
            elif (point_x >= xA and point_x <= xB) and (point_y >= yA and point_y <= yA + (yB - yA)) and point_y >= 625:
                correct_detections += 1
                break
    accuracy = correct_detections / len(central_points)
    return accuracy


# Load a model
model = YOLO("yolov8n.pt")

# Load an image
image = cv2.imread("obrazy_wszystkie/IMG_7.jpg")
gt_path = 'gt_wszystkie/GT_IMG_7.mat'

# Start the timer
start_time = time.time()

# Get results from the model
results = model(image, classes=0)

# End the timer
end_time = time.time()

# Calculate the detection time
detection_time = end_time - start_time
print("Detection Time: ", detection_time, "seconds")

boxes = []

# Draw bounding boxes on the image
for result in results:
    for box in result.boxes.xyxy:
        x1, y1, x2, y2 = box
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        boxes.append([x1, y1, x2, y2])

central_points = getHumanLocations(gt_path)
drawCentralPoints(image, central_points)
print("Accuracy: " + str(calculateAccuracy(boxes, central_points)))

# Display the image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
