# SE2 Code for Firebrand Detection and Analysis

This repository contains the principal Python scripts developed and updated for the **SE2 workflow**, including:

- data acquisition
- sensor synchronisation
- dataset generation
- YOLO-based model training
- automated post-processing and firebrand measurement

Collectively, these programs support the synchronised capture, detection, and quantitative analysis of firebrands described in the associated dissertation methodology.

---

## Repository overview

The codebase is organised around four main workflow stages:

1. **SE2 acquisition and synchronisation**
2. **Video conditioning and dataset construction**
3. **YOLOv8 training and evaluation**
4. **Post-processing and firebrand measurement**

---

## 1. SE2 acquisition and synchronisation scripts

These scripts are intended for deployment on the Raspberry Pi-based **SE2 platform**. They coordinate:

- RGB video capture
- GNSS timestamp and location logging
- thermocouple recording
- session directory creation for structured downstream analysis

### Main scripts

- `run.py`  
  Primary acquisition script coordinating sensor start-up, synchronised logging, and recording.

- `gps_*.py`  
  Helper routines for parsing GNSS data and generating UTC-anchored timestamps.

- `thermocouple_*.py`  
  Temperature logging routines for local thermal measurements.

- `session_*.py`  
  Utilities for directory naming, metadata export, and structured file storage.

---

## 2. Video conditioning and dataset-construction scripts

These scripts convert raw recorded video into model-ready datasets for training and validation. Main processing steps include:

- frame extraction
- spatial tiling
- file organisation
- preparation of training and validation data in YOLO format

### Main scripts

- `frame_sampling.py`  
  Extracts video frames at the selected sampling rate.

- `tiling.py`  
  Subdivides full-resolution frames into overlapping `640 × 640` tiles.

- `dataset_builder.py`  
  Organises tiled imagery and labels into the standard YOLO directory structure.

- `split_dataset.py`  
  Generates training and validation subsets.

---

## 3. YOLOv8 training and evaluation scripts

These scripts and command blocks are used to train and evaluate the firebrand detector. They support both:

- **Phase A** training on small-firebrand imagery
- **Phase B** fine-tuning on the mixed-scale dataset

### Main scripts and files

- `train_phaseA.py`  
  Trains the initial small-firebrand detector.

- `train_phaseB.py`  
  Fine-tunes the model on the mixed-scale dataset.

- `data_mixed.yaml`  
  Dataset configuration file for YOLO training.

- `evaluate.py`  
  Applies the trained detector to held-out imagery and records validation performance.

### Typical YOLO commands

Training and inference were carried out using command-line calls such as:

```bash
yolo detect train model=yolov8s.pt data=data_mixed.yaml imgsz=640 epochs=60 batch=16 device=0
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/source imgsz=640 conf=0.5 device=0 save=True
```

---

## 4. Post-processing and firebrand measurement scripts

These scripts convert raw detector outputs into quantitative firebrand measurements suitable for interpretation and plotting. This stage includes:

- tiled inference on full recordings
- spatial deduplication of overlapping detections
- reconstruction of frame timestamps
- conversion of bounding-box dimensions into apparent size metrics
- extraction of frame-level and summary statistics
- generation of plots and CSV outputs for further analysis

### Main scripts

- `videoanalysis.py`  
  Principal post-processing script for inference, deduplication, timestamp reconstruction, metric extraction, and output generation.

- `deduplicate.py`  
  Merges overlapping detections from adjacent image tiles into single full-frame detections.

- `timestamp_align.py`  
  Reconstructs frame timestamps relative to the GNSS-derived UTC anchor.

- `size_metrics.py`  
  Converts bounding-box dimensions or areas into apparent projected size metrics.

- `plot_results.py`  
  Generates plots of firebrand count, cumulative detections, rolling-mean trends, and apparent size distributions.

### Typical post-processing outputs

Depending on the script configuration, this stage can generate outputs such as:

- frame-by-frame firebrand counts
- deduplicated detection records
- reconstructed timestamps
- cumulative detection curves
- rolling-mean count plots
- apparent size histograms
- summary CSV and text files
