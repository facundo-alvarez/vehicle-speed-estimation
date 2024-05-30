from videomodule import Video
from yolov8 import YoloV8
import cv2
import numpy as np
from deepsorttracker import DeepSortTracker

video_file = 'resources/video.mp4'
output_path = 'resources/result.mp4'
video_module = Video(video_file)

target_classes = [2, 3, 5, 7]
model = YoloV8('yolov8n.pt', target_classes)

target_height = 150
target_width = 28.5

roi = np.array([[456, 204], [773, 215], [1070, 397], [146, 366]], dtype=np.float32)
target = np.array([[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]], dtype=np.float32)


tracker = DeepSortTracker(0.3)


def get_frame(frame):
    resized_frame, scale_x, scale_y = video_module.resize_image(frame, 640)
    boxes = model.process_frame(resized_frame)

    scaled_roi = roi.copy()
    scaled_roi[:, 0] = scaled_roi[:, 0] * scale_x  # Scale x coordinates
    scaled_roi[:, 1] = scaled_roi[:, 1] * scale_y  # Scale y coordinates
    scaled_roi = scaled_roi.astype(np.int32)
    frame = tracker.track(boxes, scaled_roi, resized_frame)

    cv2.line(resized_frame, (scaled_roi[0][0], scaled_roi[0][1]), (scaled_roi[1][0], scaled_roi[1][1]), (255, 0, 0), 2)
    cv2.line(resized_frame, (scaled_roi[2][0], scaled_roi[2][1]), (scaled_roi[3][0], scaled_roi[3][1]), (255, 0, 0), 2)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        video_module.remove_callback()

video_module.set_callback(get_frame)
video_module.get_frames()
cv2.destroyAllWindows()
