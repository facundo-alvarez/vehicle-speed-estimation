import cv2
import string
from typing import Callable
import numpy as np

class Video():
    def __init__(self, input_path:string, output_path:string, codec = 'mp4v') -> None:
        self.input_path = input_path
        self.cap = cv2.VideoCapture(self.input_path)
        self.callback = None
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.out = cv2.VideoWriter(
            output_path, 
            fourcc,
            self.cap.get(cv2.CAP_PROP_FPS), 
            (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    def set_callback(self, callback: Callable[[any], None]) -> None:
        self.callback = callback

    def remove_callback(self) -> None:
        self.callback = None

    def dispose(self) -> None:
        self.cap.release()
        self.out.release()

    @staticmethod
    def resize_image(image, target_size):
        h, w, _ = image.shape
        scale = min(target_size / h, target_size / w)
        nh, nw = int(h * scale), int(w * scale)
        resized_image = cv2.resize(image, (nw, nh))
        
        # Create a new image of the target size and fill it with zeros (black)
        new_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        new_image[:nh, :nw, :] = resized_image
        
        # Calculate the scale factors for x and y axis
        scale_x = nw / w
        scale_y = nh / h
        
        return new_image, scale_x, scale_y
    
    def get_fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_frames(self) -> None:
        if not self.cap.isOpened():
            print("Error: Unable to open video file.")
            return
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            if self.callback is None:
                break
            
            self.callback(frame)

    def save_frames(self, frame):
        self.out.write(frame)

