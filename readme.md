# Vehicle Speed Estimation

Real-time vehicle detection, tracking, and speed estimation from video. The system
detects vehicles in each frame, tracks them across frames, and estimates their speed
in km/h by mapping image coordinates to real-world distances using a perspective
transform.

## Features

- **Swappable detection backends** — YOLOv8, Faster R-CNN, and FCOS are interchangeable
  behind a common interface, so detectors can be compared without changing the pipeline.
- **Multi-object tracking** with DeepSORT, including region-of-interest (ROI) filtering
  so only vehicles inside the measurement zone are tracked.
- **Real-world speed calculation** using a homography (perspective transform) from the
  camera plane to ground coordinates, with moving-average smoothing per vehicle.
- **Performance instrumentation** — total runtime and average per-frame processing time
  are measured for each detector.

## Tech stack

- Python
- OpenCV
- PyTorch / torchvision (YOLOv8, Faster R-CNN, FCOS)
- `deep-sort-realtime`
- NumPy

## Getting started

```bash
pip install -r requirements.txt
```

Place an input video at `resources/video.mp4`, then run:

```bash
python main.py
```

The annotated output is written to `resources/result.mp4`.

## How it works

1. Each frame is resized and passed to the selected detector.
2. Detections inside the ROI are handed to the DeepSORT tracker, which assigns stable IDs.
3. Each tracked vehicle's ground position is computed via the perspective transform.
4. Speed is derived from the change in ground position over time and smoothed.

## Project structure

```
main.py              # Pipeline entry point
videomodule.py       # Video I/O and frame callback handling
yolov8.py            # YOLOv8 detector
fasterrcnn.py        # Faster R-CNN detector
fcos.py              # FCOS detector
deepsorttracker.py   # DeepSORT tracking + ROI filtering
speedcalculator.py   # Per-vehicle speed estimation
```
