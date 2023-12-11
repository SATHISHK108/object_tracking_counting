import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
from deepsort_tracker import*

# Initialize YOLO models
model = YOLO("D://sathish//tyre//yolov8n_seg_e20.pt")
model_1 = YOLO("D://sathish//tyre//ok_ceat_ok_bridgestone_5_22pm.pt")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Device:", device)

model.to(device)
model_1.to(device)

cap = cv2.VideoCapture('D://sathish//tyre//tyre.mp4')
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize variables for FPS calculation
fps_update_interval = 1  # Update FPS every 1 second
start_time = time.time()
frame_count = 0

count=0
tracker=Tracker()

line_y = 480
offset = 10
track_id_list = []
track_id_list = []
detection_threshold = 0.3
while True:
    ret, img = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break
    count += 1
    if count % 2 != 0:
        continue

    img = cv2.resize(img,(480,720))
    img = cv2.rotate(img, cv2.ROTATE_180)
    #tracker_class_id dimenssions
    list=[]
    #calibration based on camera distance(sum of pixels from all four side of square,
    # here we assume side as 20cm,
    # in that 20cm we have 147.5 pixels)
    perimeter = 590 #147.5 * 4
    # Pixel to cm ratio
    pixel_cm_ratio = perimeter / 20 # 20 - per side length in cm

    # Get YOLO predictions
    result = model(img)
    result1 = model_1(img)

    boxes = result[0].boxes.numpy()
    boxes1 = result1[0].boxes.numpy()

    if boxes1.shape[0] > 0:
        boxes1.data[0][5] = 2  # Set the class index to 2 for boxes1
        result_concate = np.concatenate((boxes.data, boxes1.data), axis = 0)
    else:
        result_concate = boxes.data

    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - start_time

    # Calculate FPS
    if elapsed_time >= fps_update_interval:
        fps = frame_count / elapsed_time
        start_time = current_time
        frame_count = 0

    # Draw bounding boxes and display information
    for prediction in result_concate:
        detections = []
        print('prediction', prediction)
        xmin, ymin, xmax, ymax, confidence, class_idx = prediction
        x1, y1, x2, y2 = map(int, [xmin, ymin, xmax, ymax])
        w = x2 - x1
        h = y2 - y1
        if class_idx == 2:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 246), 2)

        if class_idx in [0, 1]:
            # Calculate object dimensions
            object_width = w / pixel_cm_ratio
            object_height = h / pixel_cm_ratio
            if class_idx == 0:
                cv2.rectangle(img, (x1, y1), (x2, y2), (208, 59, 22), 2)
            if class_idx == 1:
                cv2.rectangle(img, (x1, y1), (x2, y2), (64, 255, 0), 2)
            # Draw dimensions
            cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (x1, y1 - 20), cv2.FONT_HERSHEY_PLAIN, 1, (208, 152, 22), 2)
            cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (x2, y2 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (208, 152, 22), 2)

            if class_idx == 0:
                class_id = int(class_idx)
                if confidence > detection_threshold:
                    detections.append([x1, y1, x2, y2, confidence])

                    print('detections', detections)
                    tracker.update(img, detections)
                    #print('objects_bbs_ids', objects_bbs_ids)
                    print('tracker.tracks ', tracker.tracks)
                    for track in tracker.tracks:
                        bbox = track.bbox
                        x3, y3, x4, y4 = bbox
                        track_id = track.track_id
                        print('track_id', track_id)
                        #     list.append([x1,y1,x2,y2])
                        # bbox_id=tracker.update(list)
                        # for bbox in bbox_id:
                            # x3,y3,x4,y4,id=bbox
                        cx=int(x3+x4)//2
                        cy=int(y3+y4)//2
                        print('cy ', cy)
                        print()
                        # cv2.circle(img,(cx,cy),4,(0,0,255),-1)
                        # cv2.putText(img,str(track_id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                        if line_y < (cy + offset) and line_y > (cy - offset):
                            if track_id not in track_id_list:
                                track_id_list.append(track_id)

                            # cv2.circle(img,(cx,cy),4,(0,0,255),-1)
                            # cv2.putText(img,str(track_id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)


        # Draw bounding box
        #cv2.rectangle(img, (x1, y1), (x2, y2), (208, 59, 22), 2)

    # Display FPS on top left corner
    cv2.putText(img, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.putText(img, "count: {}".format(len(track_id_list)), (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.line(img,(0,line_y),(1920,line_y),(255,255,255),2)

    print(track_id_list)
    # Display the captured frame
    cv2.imshow('Webcam Feed', img)

    # Break the loop when 'Esc' key (ASCII code 27) is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
