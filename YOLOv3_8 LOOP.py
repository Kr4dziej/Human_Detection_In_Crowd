from ultralytics import YOLO
import cv2
import numpy as np
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

# Ścieżka do folderu z obrazami i plikami GT
image_folder = "obrazy_wszystkie"
gt_folder = "gt_wszystkie"

# Ścieżka do pliku JSON, do którego zapisywane będą wyniki
json_path = "wyniki_json/YOLOv8_tescik.json"

# Ścieżka do folderu, w którym będą zapisywane obrazy wynikowe
output_folder = "wyniki_jpg/YOLOv8zxc"

# Inicjalizacja pustego słownika do przechowywania wyników
wyniki = {}

# Ścieżka do nowego pliku JSON, do którego zapisywane będą wyniki detekcji
detection_results_json_path = "wyniki_json/YOLOv8_sylwetki_yest.json"

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

        # Get results from the model
        results = model(image, classes=0)

        # Zakończenie pomiaru czasu
        end_time = time.time()

        # Obliczenie czasu detekcji
        detection_time = end_time - start_time

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
        accuracy = calculateAccuracy(boxes, central_points)
        human_number = getHumanNumber(gt_path)
        detection_results = calculateDetectionResults(boxes, central_points)

        # Zapisanie wyników do słownika
        wyniki[filename] = {
            "detection_time": detection_time,
            "accuracy": accuracy,
            "human_number": human_number
        }

        # Zapisanie obrazu wynikowego do folderu wynikowego
        #output_path = os.path.join(output_folder, filename)
        #cv2.imwrite(output_path, image)

        # Zapisanie wyników detekcji poszczególnych sylwetek
        detection_results_dict[filename] = detection_results


# Zapisanie wyników do pliku JSON
with open(json_path, 'w') as f:
    json.dump(wyniki, f)

# Zapisanie wyników detekcji do nowego pliku JSON
with open(detection_results_json_path, 'w') as f:
    json.dump(detection_results_dict, f)