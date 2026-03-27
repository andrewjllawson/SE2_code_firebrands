"""

Updated on Mar 25 2026

@author: Andrew Lawson

Train Phase A: initial small-firebrand detector

"""

from ultralytics import YOLO

# Load the default YOLOv8 nano pretrained model
model = YOLO("yolov8n.pt")

# Train on the Phase A dataset
model.train(
    data="data_new.yaml",
    imgsz=640,
    epochs=60,
    batch=16,
    device=0
)