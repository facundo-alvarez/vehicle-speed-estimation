from torchvision.io.video import read_video
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights, ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.ops import nms
import cv2

# Step 1: Initialize model with the best available weights
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.eval()

frames, _, _ = read_video("resources/example.mp4")
frames = frames.permute(0, 3, 1, 2)

for img in frames:
    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = [preprocess(img)]

    # Perform inference on each image in the batch
    prediction = model(batch)[0]  # unsqueeze to add batch dimension

    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                            labels=labels,
                            colors="red",
                            width=4, font_size=30)

    im = to_pil_image(box.detach())
    im.show()