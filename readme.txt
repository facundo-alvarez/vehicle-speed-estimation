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


video:
https://pixabay.com/es/videos/carretera-tr%C3%A1fico-vehiculos-carros-56310/

sizes:
https://www.german-autobahn.eu/index.asp?page=design#:~:text=Two%2C%20three%2C%20or%20occasionally%20four,lanes%20are%203.5%20meters%20wide.

Location:
https://www.google.com/maps/@48.7366386,11.4677973,3a,75y,138.61h,80.95t/data=!3m7!1e1!3m5!1sTtrBI07dWNcI00VhxpjHJQ!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fpanoid%3DTtrBI07dWNcI00VhxpjHJQ%26cb_client%3Dmaps_sv.tactile.gps%26w%3D203%26h%3D100%26yaw%3D218.97879%26pitch%3D0%26thumbfov%3D100!7i16384!8i8192?coh=205409&entry=ttu
