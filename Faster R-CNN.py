import torch
import torchvision
from torchvision import transforms as T
import cv2
import time
from scipy.io import loadmat
import os
import json


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


# Load the pre-trained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load an image
image = cv2.imread("obrazy_wszystkie/IMG_7.jpg")
gt_path = 'gt_wszystkie/GT_IMG_7.mat'

# Transform the image
transform = T.Compose([T.ToTensor()])
img = transform(image)

# Start the timer
start_time = time.time()

# Perform the object detection
with torch.no_grad():
    prediction = model([img])

# End the timer
end_time = time.time()

# Calculate the detection time
detection_time = end_time - start_time
print("Detection Time: ", detection_time, "seconds")

# Extract bounding boxes, labels, and scores
boxes, labels, scores = prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']

# Convert the image tensor to a numpy array
img = img.mul(255).permute(1, 2, 0).byte().numpy()

# Convert the numpy image back to cv::Mat
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Apply non-maxima suppression to suppress weak, overlapping bounding boxes
indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.2, nms_threshold=0.9)
NMSBoxes = []

# Draw bounding boxes on the image
for i in indices:
    # Only consider bounding boxes of class 1 (human) with a confidence score above a certain threshold
    if labels[i] == 1 and scores[i] > 0.5:
        xmin, ymin, xmax, ymax = map(int, boxes[i].tolist())
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        NMSBoxes.append([xmin, ymin, xmax, ymax])

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
central_points = getHumanLocations(gt_path)
drawCentralPoints(img, central_points)
print("Accuracy: " + str(calculateAccuracy(NMSBoxes, central_points)))
print(getHumanNumber(gt_path))
print("Detection Results: " + str(calculateDetectionResults(NMSBoxes, central_points)))

# Display the image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
