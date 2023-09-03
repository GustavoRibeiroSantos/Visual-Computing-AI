import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox  
from tracker import *
import numpy as np

def func_iou(box1, box2):
    # Calculate the intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate the union area
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union = area1 + area2 - intersection

    # Calculate the IoU
    iou = intersection / union

    return iou

cap = cv2.VideoCapture("security.mp4") # YouTube Video URL as input

# initialize variables
count = 0
frame_count = 0
prev_boxes = []
prev_box_ids = []
person_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    roi = frame[60: 352,80:540]

    frame=cv2.resize(roi,(1020,600))
    bbox, label, conf = cv.detect_common_objects(frame, model='yolov4',confidence=0.4)

    current_people_bbox = []
    current_people_label = []
    current_people_conf = []
    # loop through detected boxes
    for i, box in enumerate(bbox):
        
        # check if person
        if label[i] == 'person':
            current_people_bbox.append(bbox[i])
            current_people_label.append(label[i])
            current_people_conf.append(conf[i])
            # calculate iou with previous boxes
            ious = []
            for prev_box in prev_boxes:
                iou = func_iou(box, prev_box)
                ious.append(iou)
                
            # check if box matches previous box
            if ious and max(ious) > 0.38:
                idx = np.argmax(ious)
                prev_boxes[idx] = box
                person_id = prev_box_ids[idx]
            else:
                # assign new ID to person
                person_count += 1
                person_id = person_count
                
                # save box and ID
                prev_boxes.append(box)
                prev_box_ids.append(person_id)
            

            # draw box and ID on frame
            draw_bbox(frame, current_people_bbox, current_people_label,conf)
    cv2.putText(frame, 'Total of person: {}'.format(person_count), (50, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
    
    cv2.imshow("FRAME",frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

print(person_count)


cap.release()
cv2.destroyAllWindows()
