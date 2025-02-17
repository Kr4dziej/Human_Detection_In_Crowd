from imutils.object_detection import non_max_suppression
from scipy.io import loadmat
import numpy as np
import cv2
import imutils
import time
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

# HOG descriptor and person detector initialization
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Ścieżka do folderu z obrazami i plikami GT
image_folder = "obrazy_wszystkie"
gt_folder = "gt_wszystkie"

# Ścieżka do pliku JSON, do którego zapisywane będą wyniki
json_path = "wyniki_json/HOG.json"

# Ścieżka do folderu, w którym będą zapisywane obrazy wynikowe
output_folder = "wyniki_jpg/HOG"

# Inicjalizacja pustego słownika do przechowywania wyników
wyniki = {}

# Ścieżka do nowego pliku JSON, do którego zapisywane będą wyniki detekcji
detection_results_json_path = "wyniki_json/HOG_sylwetki.json"

# Inicjalizacja pustego słownika do przechowywania wyników detekcji
detection_results_dict = {}

# Przejście przez wszystkie pliki w folderze z obrazami
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg"):
        # Wczytanie obrazu
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)

        # Wczytanie odpowiadającego pliku GT
        gt_filename = "GT_" + filename.split('.')[0] + ".mat"
        gt_path = os.path.join(gt_folder, gt_filename)

        # Rozpoczęcie pomiaru czasu
        start_time = time.time()

        # people detector, returning bounding boxes and confidence
        boxes, weights = hog.detectMultiScale(image, winStride=(8, 8))

        # End the timer
        end_time = time.time()

        # Obliczenie czasu detekcji
        detection_time = end_time - start_time

        # creating array with rectangle vertexes
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        # non max suppression function
        picked_boxes = non_max_suppression(boxes, probs=None, overlapThresh=0.6)
        # picked_boxes = boxes

        # deleting redundant weights
        k = -1
        for i, box in enumerate(boxes):
            k = k + 1
            if box not in picked_boxes:
                weights = np.delete(weights, k)
                k = k - 1

        # drawing rectangles
        for i, (xA, yA, xB, yB) in enumerate(picked_boxes):
            # display the detected boxes in the colour picture
            #if weights[i] < 0.3:
            #    continue
            #elif weights[i] > 0.3:
                cx = int((xA + xB) / 2)
                cy = int((yA + yB) / 2)
                cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

        central_points = getHumanLocations(gt_path)
        drawCentralPoints(image, central_points)
        accuracy = calculateAccuracy(picked_boxes, central_points)
        human_number = getHumanNumber(gt_path)
        detection_results = calculateDetectionResults(picked_boxes, central_points)

        # Zapisanie wyników do słownika
        wyniki[filename] = {
            "detection_time": detection_time,
            "accuracy": accuracy,
            "human_number": human_number
        }

        # Zapisanie obrazu wynikowego do folderu wynikowego
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image)

        # Zapisanie wyników detekcji poszczególnych sylwetek
        #detection_results_dict[filename] = detection_results


# Zapisanie wyników do pliku JSON
#with open(json_path, 'w') as f:
#    json.dump(wyniki, f)

# Zapisanie wyników detekcji do nowego pliku JSON
#with open(detection_results_json_path, 'w') as f:
#    json.dump(detection_results_dict, f)
