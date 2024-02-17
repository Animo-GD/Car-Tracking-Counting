import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# COCO Classes
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']

# Tracking
tracker = Sort(max_age=20, min_hits=2)
# Limits For The Counter Line
limits = [640,970,1700,610]
# Car Id To Count The Cars
car_ids = []
# Counter Image
counter_img = cv2.imread("../Images/counter.png",cv2.IMREAD_UNCHANGED)
# Setup The Video
cap = cv2.VideoCapture("../Videos/race.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("result_video.mp4",fourcc,30,(1920,1080))
# Mask To Make The Detection In a Specific Region.
mask = cv2.imread("../Images/mask.png")
# Getting The Model.
model = YOLO("../weights/yolov8n.pt")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # The Region We Want To Detect
    ImgRegion = cv2.bitwise_and(frame, mask)

    detections = np.empty((0, 5))
    # Drawing The Counter Line
    cv2.line(frame,(limits[0],limits[1]),(limits[2],limits[3]),(255,0,0),4)
    # Counter Box

    cvzone.overlayPNG(frame,counter_img)
    cv2.putText(frame, f"{len(car_ids)}", (140, 55),cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),3)
    # result contain the bounding boxes of each frame
    results = model(ImgRegion, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil(box.conf[0] * 100) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classes[cls]

            if currentClass == "car" and conf > 0.3:
                # cvzone.cornerRect(frame, (x1, y1, w, h), l=10)
                # cvzone.putTextRect(frame, f"{currentClass} {conf}"
                #                    , (max(0, x1), max(30, y1)),
                #                    scale=2,
                #                    thickness=2,
                #                    offset=5)

                # Getting The Bounding Boxes To Give IDs
                bounding_box = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, bounding_box))
    resultsTracker = tracker.update(detections)
    for result in resultsTracker:
        x1, y1, x2, y2, ID = map(int, result)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h), l=10)
        cvzone.putTextRect(frame, f"ID: {ID}"
                           , (max(0, x1), max(30, y1)),
                           scale=2,
                           thickness=2,
                           offset=5)
        # Drawing The Center Of the Car.
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

        # If The Center Cross The Line We Will Count 1
           if (limits[0] < cx < limits[2]) and (limits[3]-30 < cy < limits[1]+30) and car_ids.count(ID)==0:
                      car_ids.append(ID)
                      cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 4)

    out.write(frame)
    cv2.imshow("cam", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindow()
