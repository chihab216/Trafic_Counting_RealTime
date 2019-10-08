import cv2


def get_bounding_boxes(frame):
    fullbody_cascade = cv2.CascadeClassifier('./HaarCascades/car.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bounding_boxes = fullbody_cascade.detectMultiScale(gray)
    bounding_boxes=list( bounding_boxes)
    fullbody_cascadescooter = cv2.CascadeClassifier("C:/Users/Lenovo/Desktop/Vehicle-Counting-master/HaarCascades/scooter.xml")
    bounding_boxes2 = fullbody_cascadescooter.detectMultiScale(gray)
    bounding_boxes2=list( bounding_boxes2)
    bb=[]
    for boxes in bounding_boxes:
        boxes=list(boxes)
        boxes.append(1)
        bb.append(boxes)
    for boxes in bounding_boxes2:
        boxes=list(boxes)
        boxes.append(3)
        bb.append(boxes)   
    
    return bb