# Quick Start Guide

This guide will help you get started with the Computer Vision Engineer Assignment solutions.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- 10GB+ free disk space

## Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Task 1: Custom Object Detection

### Setup

1. Prepare your dataset in COCO format:
```bash
cd task1_object_detection
python scripts/download_dataset.py  # Creates directory structure
```

2. Place your images in `data/train/images/` and `data/val/images/`
3. Create annotation files `data/train/annotations.json` and `data/val/annotations.json` in COCO format

### Training

```bash
python train.py --config configs/default.yaml
```

### Evaluation

```bash
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_model.pth
```

### Inference

```bash
# Single image
python inference.py --model checkpoints/best_model.pth --input test_image.jpg --output results/

# Directory of images
python inference.py --model checkpoints/best_model.pth --input test_images/ --output results/

# Video
python inference.py --model checkpoints/best_model.pth --input video.mp4 --output output_video.mp4
```

## Task 2: Quality Inspection System

### Setup

1. Prepare your dataset with defect annotations:
```bash
cd task2_quality_inspection
```

2. Place images in `data/train/images/` and `data/val/images/`
3. Create annotation files in COCO format (see `samples/README.md` for format)

### Training

```bash
python train_inspection_model.py --config config.yaml
```

### Inference

```bash
# Single image
python inspect.py --model checkpoints/best_model.pth --input sample.jpg --output results/

# Directory
python inspect.py --model checkpoints/best_model.pth --input images/ --output results/
```

The output will include:
- Annotated images with bounding boxes
- JSON files with defect details including:
  - Defect type and confidence
  - Bounding box coordinates
  - Center pixel coordinates (x, y)
  - Severity assessment

## Task 3: VLM Design Document

The comprehensive design document is located at:
```
task3_vlm_design/VLM_Design_Document.md
```

This document addresses:
- Model selection and rationale
- Architecture design for PCB inspection
- Optimization strategies for <2s inference
- Hallucination mitigation techniques
- Multi-stage training plan
- Validation methodology

## Project Structure

```
computer_vision/
├── task1_object_detection/      # Custom Faster R-CNN implementation
│   ├── models/                   # Model architectures
│   ├── utils/                    # Dataset, losses, metrics
│   ├── configs/                  # Configuration files
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   ├── inference.py              # Inference script
│   └── REPORT.md                 # Detailed report
│
├── task2_quality_inspection/     # Defect detection system
│   ├── models/                   # Defect detection models
│   ├── utils/                    # Dataset utilities
│   ├── samples/                  # Sample images and annotations
│   ├── inspect.py                # Main inspection script
│   ├── train_inspection_model.py # Training script
│   └── README.md                 # Task 2 documentation
│
├── task3_vlm_design/             # VLM design document
│   └── VLM_Design_Document.md    # Comprehensive design document
│
├── requirements.txt              # Python dependencies
├── README.md                     # Main README
└── QUICKSTART.md                 # This file
```

## Notes

- **Training from Scratch**: Task 1 models are trained from scratch (no pre-trained weights)
- **Dataset Format**: Both tasks use COCO annotation format
- **GPU Recommended**: Training is much faster on GPU, but CPU inference is possible
- **Model Checkpoints**: Trained models are saved in `checkpoints/` directories
- **Results**: Evaluation results and annotated images are saved in `results/` directories

## Troubleshooting

### Out of Memory
- Reduce batch size in config files
- Use gradient accumulation
- Reduce image resolution

### Slow Training
- Use GPU if available
- Reduce number of workers if CPU-bound
- Enable mixed precision training (add to training scripts)

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+)
- Verify you're in the correct directory

## Next Steps

1. Review the detailed reports in each task directory
2. Customize configurations for your specific use case
3. Train models on your datasets
4. Evaluate and iterate on model performance

## Support

For questions or issues, refer to:
- Task 1: `task1_object_detection/REPORT.md`
- Task 2: `task2_quality_inspection/README.md`
- Task 3: `task3_vlm_design/VLM_Design_Document.md`



