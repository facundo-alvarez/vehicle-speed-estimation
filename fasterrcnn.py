from objectdetector import ObjectDetector
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image
import numpy as np
import time

class FasterRCNN(ObjectDetector):
    def __init__(self, target_classes:list) -> None:
        super().__init__(np.array(target_classes) + 1)
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights, box_score_thresh=0.5, box_nms_thresh=0.5)
        self.model.eval()

    def process_frame(self, frame, scale_x:float, scale_y:float) -> list:
        self.count += 1
        start_time = time.time()
        preprocess = self.weights.transforms()
        img = [preprocess(Image.fromarray(frame))]
        results = self.model(img)[0]
        end_time = time.time()
        self.processTime += end_time - start_time
        filter_boxes = []
        for i in range(len(results['boxes'])):
            
            cls = results['labels'][i].item()
            class_label = self.weights.meta["categories"][cls]
            
            if cls not in self.target_classes:
                continue

            box = results['boxes'][i].detach().numpy()
            conf = results['scores'][i].item()

            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min

            # Convert to xywh format
            box_xywh = [x_min / scale_x, y_min / scale_y, width / scale_x, height / scale_y]

            filter_boxes.append((box_xywh, conf, class_label))
        
        return filter_boxes
