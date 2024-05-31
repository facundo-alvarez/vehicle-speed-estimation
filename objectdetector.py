from abc import ABC, abstractmethod

class ObjectDetector(ABC):
    def __init__(self, target_classes:list) -> None:
        self.target_classes = target_classes

    @abstractmethod
    def process_frame(self, frame, scale_x:float, scale_y:float) -> list:
        pass