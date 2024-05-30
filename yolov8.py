import string
from ultralytics import YOLO
import numpy as np

class YoloV8():
    def __init__(self, model:string, target_classes:list) -> None:
        self.model = YOLO(model)
        self.target_classes = target_classes

    def process_frame(self, frame, scale_x, scale_y) -> list:
        results = self.model.predict(frame, stream=True)
        result = next(results)
        filter_boxes = [box.numpy() for box in result.boxes if box not in self.target_classes]
        return [
            (
                [
                    (box.xywh[0][0] - box.xywh[0][2] * 0.5) / scale_x, # x coordinate
                    (box.xywh[0][1] - box.xywh[0][3] * 0.5) / scale_y, # y coordinate
                    box.xywh[0][2] / scale_x,                          # width
                    box.xywh[0][3] / scale_y                           # height
                ], 
                box.conf[0],                                           # confidence
                box.cls[0]                                             # class
            ) 
            for box in filter_boxes
]

