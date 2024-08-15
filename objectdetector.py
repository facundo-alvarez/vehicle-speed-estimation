from abc import ABC, abstractmethod

class ObjectDetector(ABC):
    def __init__(self, target_classes:list) -> None:
        self.processTime = 0.0
        self.count = 0
        self.target_classes = target_classes

    @abstractmethod
    def process_frame(self, frame, scale_x:float, scale_y:float) -> list:
        pass

        
    def get_processing_time_avg(self) -> float:
        return self.processTime / self.count