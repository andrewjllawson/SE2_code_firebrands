"""

Updated on Mar 25 2026

@author: Andrew Lawson

Train Phase B: fine-tune the detector on the mixed-scale dataset

"""

from ultralytics import YOLO

# Load the best weights from the previous training run
model = YOLO("runs/detect/train5/weights/best.pt")

# Continue training on the mixed dataset
model.train(
    data="data_mixed.yaml",
    imgsz=640,
    epochs=60,
    batch=16,
    device=0
)