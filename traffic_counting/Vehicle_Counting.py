import cv2
from trackers.tracker import create_blob, add_new_blobs, remove_duplicates
import numpy as np
from collections import OrderedDict
from detectors.detector import get_bounding_boxes
import uuid
import os
import contextlib
from datetime import datetime
import argparse
from utils.detection_roi import get_roi_frame, draw_roi
from counter import get_counting_line, is_passed_counting_line
import time

# parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('video', help='relative/absolute path to video or camera input of traffic scene')
parser.add_argument('--iscam', action='store_true', help='specify if video capture is from a camera')
parser.add_argument('--droi', help='specify a detection region of interest (ROI) \
                    i.e a set of vertices that represent the area (polygon) \
                    where you want detections to be made (format: 1,2|3,4|5,6|7,8|9,10 \
                    default: 0,0|frame_width,0|frame_width,frame_height|0,frame_height \
                    [i.e the whole video frame])')
parser.add_argument('--showdroi', action='store_true', help='display/overlay the detection roi on the video')
parser.add_argument('--mctf', type=int, help='maximum consecutive tracking failures \
                    i.e number of tracking failures before the tracker concludes \
                    the tracked object has left the frame')
parser.add_argument('--di', type=int, help='detection interval i.e number of frames \
                    before detection is carried out again (in order to find new vehicles \
                    and update the trackers of old ones)')
parser.add_argument('--detector', help='select a model/algorithm to use for vehicle detection \
                    (options: yolo, haarc, bgsub, ssd | default: yolo)')
parser.add_argument('--tracker', help='select a model/algorithm to use for vehicle tracking \
                    (options: csrt, kcf, camshift | default: kcf)')
parser.add_argument('--record', action='store_true', help='record video and vehicle count logs')
parser.add_argument('--clposition', help='position of counting line (options: top, bottom, \
                    left, right | default: bottom)')
args = parser.parse_args()


# capture traffic scene video
video = int(args.video) if args.iscam else args.video
cap = cv2.VideoCapture(video)
_, frame = cap.read()
classes_of_interest = ['bicycle', 'car', 'bus', 'motorcycle', 'truck','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant','pasdedant']
    
# configs
blobs = OrderedDict()
blob_id = 1
frame_counter = 0
DETECTION_INTERVAL = 10 if args.di == None else args.di
MAX_CONSECUTIVE_TRACKING_FAILURES = 3 if args.mctf == None else args.mctf
detector = 'yolo' if args.detector == None else args.detector
tracker = 'kcf' if args.tracker == None else args.tracker
f_height, f_width, _ = frame.shape

# init video object and log file to record counting
if args.record:
    output_video = cv2.VideoWriter('./videos/output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (f_width, f_height))

    log_file_name = 'log.txt'
    with contextlib.suppress(FileNotFoundError):
        os.remove(log_file_name)
    log_file = open(log_file_name, 'a')
    log_file.write('vehicle_id, count, datetime\n')
    log_file.flush()

# set counting line
clposition = 'bottom' if args.clposition == None else args.clposition
counting_line = get_counting_line(clposition, f_width, f_height)
vehicle_count = 0
bicycle_count = 0
car_count = 0
motorcycle_count = 0
bus_count = 0
truck_count = 0

# create detection ROI
droi = [(0, 0), (f_width, 0), (f_width, f_height), (0, f_height)]
if args.droi:
    droi = []
    points = args.droi.replace(' ', '').split('|')
    for point_str in points:
        point = tuple(map(int, point_str.split(',')))
        droi.append(point)

# initialize trackers and create new blobs
droi_frame = get_roi_frame(frame, droi)
initial_bboxes = get_bounding_boxes(droi_frame, detector)
for box in initial_bboxes:
    print(box)
    _blob = create_blob(box, frame, tracker)
    blobs[blob_id] = _blob
    blob_id += 1
    
n=0
start = time.time()
while True:
    
    k = cv2.waitKey(1)
    n+=1
    if args.iscam or cap.get(cv2.CAP_PROP_POS_FRAMES) + 1 < cap.get(cv2.CAP_PROP_FRAME_COUNT):
        _, frame = cap.read()
        if n%1==0:
            for _id, blob in list(blobs.items()):
                Bid=blob.classID
                # update trackers
                success, box = blob.tracker.update(frame)
                box=list(box)
                box.append(Bid)
                print(box)
                if success:
                    blob.num_consecutive_tracking_failures = 0
                    blob.update(box)
                else:
                    blob.num_consecutive_tracking_failures += 1

                # delete untracked blobs
                if blob.num_consecutive_tracking_failures >= MAX_CONSECUTIVE_TRACKING_FAILURES:
                    del blobs[_id]

                # count vehicles
                if is_passed_counting_line(blob.centroid, counting_line, clposition) and not blob.counted:
                    blob.counted = True
                    if blob.classID == 0:
                        bicycle_count += 1
                    if blob.classID == 1:
                        car_count += 1
                    if blob.classID == 2:
                        motorcycle_count += 1
                    if blob.classID == 3:
                        bus_count += 1
                    if blob.classID == 4:
                        truck_count += 1
                    # log count data to a file (vehicle_id, count, datetime)
                    if args.record:
                        _row = '{0}, {1}, {2}, {3}, {4}, {5}, {6}\n'.format('v_' + str(_id), bicycle_count, car_count, motorcycle_count, bus_count, truck_count, datetime.now())
                        log_file.write(_row)
                        log_file.flush()

            if frame_counter >= DETECTION_INTERVAL:
                # rerun detection
                droi_frame = get_roi_frame(frame, droi)
                boxes = get_bounding_boxes(droi_frame, detector)
                blobs, current_blob_id = add_new_blobs(boxes, blobs, frame, tracker, blob_id, counting_line, clposition)
                blob_id = current_blob_id
                blobs = remove_duplicates(blobs)
                frame_counter = 0

            # draw and label blob bounding boxes
            for _id, blob in blobs.items():
                (x, y, w, h,classID) = [int(v) for v in blob.bounding_box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'v_' + str(_id)+classes_of_interest[blob.classID] , (x, y - 2), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # draw counting line
            cv2.line(frame, counting_line[0], counting_line[1], (0, 255, 0), 3)

            # display vehicle count
            cv2.putText(frame, 'car: ' + str(car_count), (20, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame,'Bus: ' + str(motorcycle_count), (20, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame,  'Motorbike: ' + str(bus_count), (20, 140), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Bicycle: ' + str(bicycle_count), (20, 180), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Truck: ' + str(truck_count), (20, 220), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
            # show detection roi
            if args.showdroi:
                frame = draw_roi(frame, droi)

            # save frame in video output
            if args.record:
                output_video.write(frame)

            # visualize vehicle counting
            resized_frame = cv2.resize(frame, (858, 480))
            cv2.imshow('tracking', resized_frame)

            frame_counter += 1

            # save frame if 's' key is pressed
            if k & 0xFF == ord('s'):
                cv2.imwrite(os.path.join('screenshots', 'ss_' + uuid.uuid4().hex + '.png'), frame)
                print('Screenshot taken.')
    else:
        end = time.time()
        seconds = end - start
        print('End of video.')
        print(seconds/frame_counter)
       # end video loop if on the last frame
        break

    # end video loop if 'q' key is pressed
    if k & 0xFF == ord('q'):
        end = time.time()
        seconds = end - start
        print('Video exited.')
        print(seconds/frame_counter)
        break

# end capture, close window, close log file and video objects if any
cap.release()
cv2.destroyAllWindows()
if args.record:
    log_file.close()
    output_video.release()