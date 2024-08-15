import string
from ultralytics import YOLO
from objectdetector import ObjectDetector
import time

class YoloV8(ObjectDetector):
    def __init__(self, model:string, target_classes:list) -> None:
        super().__init__(target_classes)
        self.model = YOLO(model)


    def process_frame(self, frame, scale_x:float, scale_y:float) -> list:
        start_time = time.time()
        results = self.model.predict(frame, stream=True)
        result = next(results)
        end_time = time.time()
        self.processTime += end_time - start_time
        self.count += 1
        filter_boxes = [box.numpy() for box in result.boxes if box not in self.target_classes and box.conf[0] >= 0.5]
        return [
            (
                [
                    (box.xywh[0][0] - box.xywh[0][2] * 0.5) / scale_x, # x coordinate
                    (box.xywh[0][1] - box.xywh[0][3] * 0.5) / scale_y, # y coordinate
                    box.xywh[0][2] / scale_x,                          # width
                    box.xywh[0][3] / scale_y                           # height
                ], 
                box.conf[0],                                           # confidence
                result.names[box.cls[0]]                               # class
            ) 
            for box in filter_boxes]