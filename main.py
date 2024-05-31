from videomodule import Video
from yolov8 import YoloV8
from fasterrcnn import FasterRCNN
from fcos import FCOS
import cv2
import numpy as np
from deepsorttracker import DeepSortTracker
from speedcalculator import SpeedCalculator

video_file = 'resources/video.mp4'
output_path = 'resources/result.mp4'
video_module = Video(video_file, output_path)

target_classes = [2, 3, 5, 7]
# model = YoloV8('yolov8n.pt', target_classes)
# model = FasterRCNN(target_classes)
model = FCOS(target_classes)

target_height = 150
target_width = 28.5

fps = int(video_module.get_fps())

roi = np.array([[456, 204], [773, 215], [1070, 397], [146, 366]], dtype=np.float32)
target = np.array([[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]], dtype=np.float32)

tracker = DeepSortTracker(0.3)
speed_calculator = SpeedCalculator()

m = cv2.getPerspectiveTransform(roi, target)

def get_frame(frame):
    
    resized_frame, scale_x, scale_y = video_module.resize_image(frame, 640)
    boxes = model.process_frame(resized_frame, scale_x, scale_y)
    boxes = sorted(boxes, key=lambda box: box[1])
    
    tracks, ids, cls, conf = tracker.track(boxes, roi, frame)
    
    overlay = frame.copy()
    overlay_alpha = 0.5

    red_color = (92, 92, 205)
    cv2.line(overlay, (int(roi[0][0]), int(roi[0][1])), (int(roi[1][0]), int(roi[1][1])), red_color, 3)
    cv2.line(overlay, (int(roi[2][0]), int(roi[2][1])), (int(roi[3][0]), int(roi[3][1])), red_color, 3)
    cv2.line(overlay, (int(roi[1][0]), int(roi[1][1])), (int(roi[2][0]), int(roi[2][1])), red_color, 3)
    cv2.line(overlay, (int(roi[0][0]), int(roi[0][1])), (int(roi[3][0]), int(roi[3][1])), red_color, 3)
    cv2.addWeighted(overlay, overlay_alpha, frame, 1 - overlay_alpha, 0, frame)
   
    orange_color = (0, 165, 255)
    for index, track in enumerate(tracks):
        tl_x = int(track[0])
        tl_y = int(track[1])
        br_x = int(track[0] + track[2] )
        br_y = int(track[1] + track[3])
        id = ids[index]
        box_cls = cls[index]
        box_conf = conf[index]

        # Calculate center coordinates
        center_x = int(tl_x + (br_x - tl_x) / 2)
        bottom_y =int(tl_y + (br_y - tl_y))

        center_point = np.array([[center_x, bottom_y]], dtype=np.float32)
        center_point = np.array([center_point]) 
        transformed_point = cv2.perspectiveTransform(center_point, m)
        transformed_center_x, transformed_bottom_y = transformed_point[0][0]

        # Calculate velocity
        speed = speed_calculator.calculate((id, transformed_center_x, transformed_bottom_y), fps)

        # Draw bounding box
        cv2.rectangle(frame, (tl_x, tl_y), (br_x, br_y), orange_color, 2)

        text_id = 'ID: ' + str(id) + ' Speed: ' + str(speed)
        text_cls_conf = 'Cls: ' + str(box_cls) + ' Conf: ' + '{0:.2f}'.format(box_conf)

        # Calculate text size
        text_size_cls_conf = cv2.getTextSize(text_cls_conf, cv2.FONT_HERSHEY_DUPLEX, 0.2, 1)[0]
        
        # Calculate rectangle dimensions
        rect_width_cls_conf = text_size_cls_conf[0]

        # Draw center point
        cv2.circle(frame, (center_x, bottom_y), 1, orange_color, 5)

        # Draw filled rectangle behind text
        cv2.rectangle(frame, (tl_x - 1, tl_y - 30), (tl_x + 38 + rect_width_cls_conf, tl_y), orange_color, -1)

        # Draw text
        cv2.putText(frame, text_id, (tl_x + 5, tl_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        cv2.putText(frame, text_cls_conf, (tl_x + 5, tl_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    
    cv2.imshow('Frame', frame)
    
    video_module.save_frames(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        video_module.remove_callback()
    
    
video_module.set_callback(get_frame)
video_module.get_frames()
cv2.destroyAllWindows()
