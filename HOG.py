from imutils.object_detection import non_max_suppression
from scipy.io import loadmat
import numpy as np
import cv2
import imutils
import time


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

# Video capturing
cap = cv2.VideoCapture('obrazy_wszystkie/IMG_1.jpg')
gt_path = 'gt_wszystkie/GT_img_1.mat'

while (cap.isOpened()):
    # Capturing frame from video
    ret, frame = cap.read()

    if ret == 0:
        break

    # resizing for faster working
    # frame = cv2.resize(frame, (640, 480))

    # Start the timer
    start_time = time.time()

    # people detector, returning bounding boxes and confidence
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    # End the timer
    end_time = time.time()

    # Calculate the detection time
    detection_time = end_time - start_time
    print("Detection Time: ", detection_time, "seconds")

    # creating array with rectangle vertexes
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    # non max suppression function
    picked_boxes = non_max_suppression(boxes, probs=None, overlapThresh=0.6)
    #picked_boxes = boxes

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

        #elif weights[i] < 0.7 and weights[i] > 0.3:
        #    cx = int((xA + xB) / 2)
        #    cy = int((yA + yB) / 2)
        #    #cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
        #    cv2.rectangle(frame, (xA, yA), (xB, yB), (50, 127, 255), 2)
        #elif weights[i] > 0.3:
            cx = int((xA + xB) / 2)
            cy = int((yA + yB) / 2)
            #cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # displaying output
    central_points = getHumanLocations(gt_path)
    drawCentralPoints(frame, central_points)
    print("Accuracy: " + str(calculateAccuracy(picked_boxes, central_points)))
    print("Detection Results: " + str(calculateDetectionResults(picked_boxes, central_points)))
    cv2.imshow('Output', frame)
    # closing video
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# realising capture and closing window
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(0)
