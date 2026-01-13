# Computer Vision Engineer Assignment

This repository contains complete solutions for three computer vision engineering tasks:

1. **Custom Object Detection with Model Training from Scratch** - Faster R-CNN implementation trained from scratch
2. **Automated Quality Inspection System for Manufacturing** - Defect detection and classification system
3. **Custom VLM Design for Industrial Quality Inspection** - Comprehensive VLM design document

## Overview

### Task 1: Custom Object Detection
- **Implementation**: Faster R-CNN from scratch (no pre-trained weights)
- **Features**: 
  - Custom CNN backbone
  - Region Proposal Network (RPN)
  - ROI Head for classification and localization
  - Training pipeline with data augmentation
  - Evaluation metrics (mAP, FPS, model size)
  - Real-time inference support
- **Report**: Detailed architecture, training methodology, and results analysis

### Task 2: Quality Inspection System
- **Implementation**: Defect detection and classification model
- **Features**:
  - Detects 4 defect types: scratches, misalignment, missing components, discoloration
  - Bounding box localization
  - Center coordinate extraction (x, y)
  - Severity assessment (High/Medium/Low)
  - JSON output with structured defect information
- **Output**: Annotated images and detailed JSON reports

### Task 3: VLM Design Document
- **Content**: Comprehensive design for Vision-Language Model for PCB inspection
- **Topics Covered**:
  - Model selection (Qwen-VL based)
  - Architecture modifications for precise localization
  - Optimization for <2s inference
  - Hallucination mitigation strategies
  - Multi-stage training plan
  - Validation methodology

## Project Structure

```
computer_vision/
├── task1_object_detection/          # Task 1: Custom Object Detection
│   ├── models/                       # Faster R-CNN model architecture
│   │   ├── faster_rcnn.py
│   │   └── __init__.py
│   ├── utils/                        # Utilities
│   │   ├── dataset.py                # Dataset loading and augmentation
│   │   ├── losses.py                  # Loss functions
│   │   └── metrics.py                 # Evaluation metrics (mAP)
│   ├── configs/                      # Configuration files
│   │   └── default.yaml
│   ├── scripts/                      # Helper scripts
│   │   └── download_dataset.py
│   ├── train.py                      # Training script
│   ├── evaluate.py                   # Evaluation script
│   ├── inference.py                  # Inference script
│   └── REPORT.md                     # Detailed report
│
├── task2_quality_inspection/        # Task 2: Quality Inspection
│   ├── models/                       # Defect detection models
│   │   ├── defect_detector.py
│   │   └── __init__.py
│   ├── utils/                        # Utilities
│   │   └── dataset.py                # Dataset utilities
│   ├── samples/                      # Sample images and annotations
│   │   └── README.md
│   ├── inspect.py                    # Main inspection script
│   ├── train_inspection_model.py     # Training script
│   ├── config.yaml                   # Configuration
│   └── README.md                     # Task 2 documentation
│
├── task3_vlm_design/                 # Task 3: VLM Design
│   └── VLM_Design_Document.md        # Comprehensive design document
│
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── QUICKSTART.md                     # Quick start guide
└── .gitignore                        # Git ignore rules
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Task 1: Object Detection

```bash
cd task1_object_detection

# Prepare dataset (creates structure)
python scripts/download_dataset.py

# Train model
python train.py --config configs/default.yaml

# Evaluate
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_model.pth

# Run inference
python inference.py --model checkpoints/best_model.pth --input test_image.jpg --output results/
```

### Task 2: Quality Inspection

```bash
cd task2_quality_inspection

# Train model
python train_inspection_model.py --config config.yaml

# Inspect images
python inspect.py --model checkpoints/best_model.pth --input sample.jpg --output results/
```

### Task 3: VLM Design

Read the comprehensive design document:
```bash
cat task3_vlm_design/VLM_Design_Document.md
```

## Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 1.12+ (with CUDA support recommended)
- **GPU**: CUDA-capable GPU recommended for training
- **RAM**: 8GB+ recommended
- **Disk Space**: 10GB+ for datasets and models

See `requirements.txt` for complete dependency list.

## Key Features

### Task 1 Features
- Custom Faster R-CNN from scratch (no pre-trained weights)
- Training pipeline with data augmentation
- Evaluation metrics: mAP, FPS, model size
- Real-time inference support
- Video processing capability
- Detailed training report

### Task 2 Features
- Defect detection and classification
- Bounding box localization
- Center coordinate extraction (x, y)
- Severity assessment
- Structured JSON output
- Batch processing support

### Task 3 Features
- Comprehensive VLM design document
- Model selection rationale
- Architecture modifications
- Optimization strategies
- Hallucination mitigation
- Training and validation plans

## Dataset Format

Both Task 1 and Task 2 use **COCO annotation format**:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 100, 200, 150],
      "area": 30000
    }
  ],
  "categories": [
    {"id": 1, "name": "class1"}
  ]
}
```

## Results and Outputs

### Task 1 Outputs
- Model checkpoints in `task1_object_detection/checkpoints/`
- Evaluation results in `task1_object_detection/results/`
- Training logs (TensorBoard) in `task1_object_detection/logs/`
- Detailed report in `task1_object_detection/REPORT.md`

### Task 2 Outputs
- Annotated images with bounding boxes
- JSON files with defect details:
  - Defect type and confidence
  - Bounding box coordinates
  - Center coordinates (x, y)
  - Severity assessment
- Results in `task2_quality_inspection/results/`

### Task 3 Output
- Comprehensive design document: `task3_vlm_design/VLM_Design_Document.md`

## Documentation

- **Main README**: This file
- **Quick Start Guide**: `QUICKSTART.md`
- **Task 1 Report**: `task1_object_detection/REPORT.md`
- **Task 2 README**: `task2_quality_inspection/README.md`
- **Task 3 Design**: `task3_vlm_design/VLM_Design_Document.md`

## Notes

- **Training from Scratch**: Task 1 models are trained without pre-trained weights
- **Platform Compatibility**: Solutions work on x86_64 and ARM platforms
- **Offline Capable**: All solutions can run offline after initial setup
- **Extensible**: Code is modular and easy to extend

