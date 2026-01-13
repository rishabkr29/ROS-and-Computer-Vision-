# Task 1: Custom Object Detection with Faster R-CNN

## Overview

This repository contains a complete implementation of Faster R-CNN for object detection, trained from scratch (no pre-trained weights). The model can detect and localize multiple object classes with bounding boxes.

## Features

- Custom Faster R-CNN implementation from scratch
- Training pipeline with data augmentation
- Evaluation metrics (mAP, FPS, model size)
- Real-time inference support
- Video processing capability
- Sample results included

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- 10GB+ free disk space

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or if using pip3:
```bash
pip3 install -r requirements.txt
```

## Quick Start

### 1. Check Sample Results (No Installation Required)

The repository includes sample results to demonstrate the output format:

```bash
# View evaluation results
cat results/evaluation_results.txt

# View training summary
cat results/training_summary.txt
```

**Sample Results Location:**
- `results/evaluation_results.txt` - Contains mAP, FPS, model size metrics
- `results/training_summary.txt` - Contains training configuration and architecture details

### 2. Prepare Dataset

The dataset should be in COCO annotation format.

```bash
# Create dataset structure
python scripts/download_dataset.py
```

This creates:
- `data/train/images/` - Training images
- `data/train/annotations.json` - Training annotations (COCO format)
- `data/val/images/` - Validation images
- `data/val/annotations.json` - Validation annotations (COCO format)

**Note:** Sample synthetic data is already included for testing.

### 3. Train the Model

```bash
python train.py --config configs/default.yaml
```

**Training Parameters:**
- Batch size: 4 (adjust in `configs/default.yaml`)
- Learning rate: 0.001
- Epochs: 50
- Model saves checkpoints to `checkpoints/`

**Monitor Training:**
```bash
# Check checkpoint creation
ls -lh checkpoints/

# View TensorBoard logs (if available)
tensorboard --logdir logs/
```

### 4. Evaluate the Model

After training, evaluate the model:

```bash
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_model.pth
```

**Output:**
- mAP (mean Average Precision)
- FPS (inference speed)
- Average inference time
- Model size
- Results saved to `results/evaluation_results.txt`

### 5. Run Inference

#### Single Image
```bash
python inference.py --model checkpoints/best_model.pth --input test_image.jpg --output results/
```

#### Directory of Images
```bash
python inference.py --model checkpoints/best_model.pth --input images/ --output results/
```

#### Video File
```bash
python inference.py --model checkpoints/best_model.pth --input video.mp4 --output output_video.mp4
```

**Output:**
- Annotated images with bounding boxes
- FPS displayed on images
- Detections saved to output directory

## Results

### Sample Results (Included)

The repository includes sample results demonstrating the output format:

1. **Evaluation Results** (`results/evaluation_results.txt`):
   ```
   mAP: 0.6523
   FPS: 12.5
   Average inference time: 80.00 ms
   Model size: 45.23 MB
   ```

2. **Training Summary** (`results/training_summary.txt`):
   - Training configuration
   - Architecture details
   - Performance metrics

### After Running Training/Evaluation

Results are saved to:
- `results/evaluation_results.txt` - Evaluation metrics
- `results/training_summary.txt` - Training details
- `results/annotated_*.jpg` - Annotated inference images

### View Results

```bash
# View evaluation metrics
cat results/evaluation_results.txt

# View training summary
cat results/training_summary.txt

# List annotated images
ls -lh results/*.jpg
```

## Dataset Format

The model uses COCO annotation format:

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
    {"id": 1, "name": "person"},
    {"id": 2, "name": "car"}
  ]
}
```

## Configuration

Edit `configs/default.yaml` to customize:
- Batch size
- Learning rate
- Number of epochs
- Number of classes
- Dataset paths

## Model Architecture

- **Backbone**: Custom CNN for feature extraction
- **RPN**: Region Proposal Network
- **ROI Head**: Classification and bounding box regression
- **Output**: Bounding boxes with class labels and confidence scores

## Troubleshooting

### Out of Memory
- Reduce batch size in `configs/default.yaml`
- Reduce image resolution
- Use gradient accumulation

### Slow Training
- Use GPU if available
- Reduce number of workers
- Enable mixed precision training

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+)
- Verify you're in the correct directory

### No Checkpoint Found
- Make sure training completed successfully
- Check `checkpoints/` directory exists
- Verify training didn't fail early

## File Structure

```
task1_object_detection/
├── models/              # Model architectures
│   ├── faster_rcnn.py
│   └── __init__.py
├── utils/               # Utilities
│   ├── dataset.py       # Dataset loading
│   ├── losses.py        # Loss functions
│   └── metrics.py      # Evaluation metrics
├── configs/             # Configuration files
│   └── default.yaml
├── scripts/             # Helper scripts
│   └── download_dataset.py
├── data/                # Dataset (not included in repo)
│   ├── train/
│   └── val/
├── checkpoints/         # Model checkpoints (not included)
├── results/             # Results and outputs
│   ├── evaluation_results.txt
│   └── training_summary.txt
├── train.py             # Training script
├── evaluate.py          # Evaluation script
├── inference.py         # Inference script
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## Performance Metrics

**Sample Results (from included sample outputs):**
- mAP: 0.6523
- FPS: 12.5
- Model size: 45.23 MB
- Average inference time: 80 ms

**Note:** Actual results will vary based on:
- Training data quality
- Training duration
- Hardware used
- Model configuration

## Citation

If you use this code, please cite:
```
Custom Faster R-CNN Implementation for Object Detection
Trained from scratch without pre-trained weights
```

