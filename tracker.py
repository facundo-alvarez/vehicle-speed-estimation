from abc import ABC, abstractmethod
import numpy as np

class Tracker(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def track(self, boxes:np.array, roi:np.array):
        pass
