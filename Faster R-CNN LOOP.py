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

# Ścieżka do folderu z obrazami i plikami GT
image_folder = "obrazy_wszystkie"
gt_folder = "gt_wszystkie"

# Ścieżka do pliku JSON, do którego zapisywane będą wyniki
json_path = "wyniki_json/Faster_R-CNN_nowe.json"

# Ścieżka do folderu, w którym będą zapisywane obrazy wynikowe
output_folder = "wyniki_jpg/Faster_R-CNN"

# Inicjalizacja pustego słownika do przechowywania wyników
results = {}

# Ścieżka do nowego pliku JSON, do którego zapisywane będą wyniki detekcji
detection_results_json_path = "wyniki_json/Faster_R-CNN_nowe_sylwetki.json"

# Inicjalizacja pustego słownika do przechowywania wyników detekcji
detection_results_dict = {}

# Transform the image
transform = T.Compose([T.ToTensor()])

# Przejście przez wszystkie pliki w folderze z obrazami
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg"):
        # Wczytanie obrazu
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)

        # Wczytanie odpowiadającego pliku GT
        gt_filename = "GT_" + filename.split('.')[0] + ".mat"
        gt_path = os.path.join(gt_folder, gt_filename)

        # Transformacja obrazu
        img = transform(image)

        # Rozpoczęcie pomiaru czasu
        start_time = time.time()

        # Wykonanie detekcji obiektów
        with torch.no_grad():
            prediction = model([img])

        # Zakończenie pomiaru czasu
        end_time = time.time()

        # Obliczenie czasu detekcji
        detection_time = end_time - start_time

        # Wyodrębnienie bounding boxes, etykiet i wyników
        boxes, labels, scores = prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']

        # Konwersja tensora obrazu na tablicę numpy
        img = img.mul(255).permute(1, 2, 0).byte().numpy()

        # Konwersja tablicy numpy z powrotem na cv::Mat
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Zastosowanie non-maxima suppression do tłumienia słabych, nakładających się na siebie bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.2, nms_threshold=0.9)
        NMSBoxes = []

        # Rysowanie bounding boxes na obrazie
        for i in indices:
            # Rozważanie tylko bounding boxes klasy 1 (człowiek) z wynikiem pewności powyżej pewnego progu
            if labels[i] == 1 and scores[i] > 0.5:
                xmin, ymin, xmax, ymax = map(int, boxes[i].tolist())
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                NMSBoxes.append([xmin, ymin, xmax, ymax])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        central_points = getHumanLocations(gt_path)
        drawCentralPoints(img, central_points)
        accuracy = calculateAccuracy(NMSBoxes, central_points)
        human_number = getHumanNumber(gt_path)
        detection_results = calculateDetectionResults(NMSBoxes, central_points)

        # Zapisanie wyników do słownika
        results[filename] = {
            "detection_time": detection_time,
            "accuracy": accuracy,
            "human_number": human_number
        }

        # Zapisanie obrazu wynikowego do folderu wynikowego
        #output_path = os.path.join(output_folder, filename)
        #cv2.imwrite(output_path, img)

        # Zapisanie wyników detekcji poszczególnych sylwetek
        detection_results_dict[filename] = detection_results


# Zapisanie wyników do pliku JSON
with open(json_path, 'w') as f:
    json.dump(results, f)

# Zapisanie wyników detekcji do nowego pliku JSON
with open(detection_results_json_path, 'w') as f:
    json.dump(detection_results_dict, f)
