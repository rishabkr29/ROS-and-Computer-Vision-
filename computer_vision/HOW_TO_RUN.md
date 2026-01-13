# How to Run Each Task

This guide explains how to run each task individually.

## Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
# or
pip3 install -r requirements.txt
```

## Quick Start - Run One Task at a Time

Use the `run_task.py` script to run tasks individually:

### Task 1: Custom Object Detection

**Demo Mode (Quick Setup):**
```bash
python3 run_task.py --task task1 --mode demo
```
This will:
- Set up the directory structure
- Generate sample synthetic data
- Show you what's ready

**Full Training Mode:**
```bash
python3 run_task.py --task task1 --mode full
```
This will:
- Train the model from scratch
- Evaluate the model (mAP, FPS, model size)
- Run inference on validation images

**Setup Only:**
```bash
python3 run_task.py --task task1 --setup-only
```

### Task 2: Quality Inspection

**Demo Mode:**
```bash
python3 run_task.py --task task2 --mode demo
```

**Full Training Mode:**
```bash
python3 run_task.py --task task2 --mode full
```

**Setup Only:**
```bash
python3 run_task.py --task task2 --setup-only
```

### Task 3: VLM Design Document

**View the document:**
```bash
python3 run_task.py --task task3
```

Or directly:
```bash
cat task3_vlm_design/VLM_Design_Document.md
```

## Manual Execution (Alternative)

If you prefer to run commands manually:

### Task 1: Manual Steps

```bash
cd task1_object_detection

# 1. Setup data structure
python3 scripts/download_dataset.py

# 2. Generate sample data (if needed)
python3 ../scripts/generate_sample_data.py --task task1

# 3. Train model
python3 train.py --config configs/default.yaml

# 4. Evaluate model
python3 evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_model.pth

# 5. Run inference
python3 inference.py --model checkpoints/best_model.pth --input data/val/images --output results/
```

### Task 2: Manual Steps

```bash
cd task2_quality_inspection

# 1. Generate sample data (if needed)
python3 ../scripts/generate_sample_data.py --task task2

# 2. Train model
python3 train_inspection_model.py --config config.yaml

# 3. Run inspection
python3 inspect.py --model checkpoints/best_model.pth --input samples/defective --output results/
```

## Results Location

After running tasks, results are saved in:

- **Task 1**: `task1_object_detection/results/`
  - Evaluation results: `evaluation_results.txt`
  - Inference images: Annotated images with bounding boxes

- **Task 2**: `task2_quality_inspection/results/`
  - JSON files with defect details
  - Annotated images with bounding boxes

- **Task 3**: `task3_vlm_design/VLM_Design_Document.md`
  - Complete design document

## Troubleshooting

### Python Command Not Found
If you get "python: command not found", use `python3` instead:
```bash
python3 run_task.py --task task1
```

### No Training Data
The script will automatically generate sample synthetic data if none is found. For real training, you should:
1. Download a real dataset (e.g., PASCAL VOC)
2. Place images in `data/train/images/` and `data/val/images/`
3. Create COCO-format annotation files

### Out of Memory
- Reduce batch size in config files
- Use smaller image resolution
- Train on CPU (slower but uses less memory)

### Model Checkpoint Not Found
If you see "checkpoint not found" errors:
- Make sure training completed successfully
- Check that `checkpoints/best_model.pth` exists
- Run training first: `python3 run_task.py --task task1 --mode full`

## Next Steps

1. **Review Results**: Check the results directories for outputs
2. **Customize**: Modify config files for your specific use case
3. **Train on Real Data**: Replace synthetic data with your actual dataset
4. **Evaluate**: Review the evaluation metrics and reports

## Notes

- **Demo Mode**: Quick setup and verification, no actual training
- **Full Mode**: Complete training, evaluation, and inference
- **Synthetic Data**: Generated data is for testing only. For real results, use actual datasets.

