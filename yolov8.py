import string
from ultralytics import YOLO
import numpy as np

class YoloV8():
    def __init__(self, model:string, target_classes:list) -> None:
        self.model = YOLO(model)
        self.target_classes = target_classes

    def process_frame(self, frame) -> np.ndarray:
        results = self.model.predict(frame, stream=True)
        result = next(results)
        return [box.numpy() for box in result.boxes if box not in self.target_classes]