# Step-by-Step Testing Guide

Follow these commands exactly to test each task.

## Prerequisites

### Step 1: Navigate to Project Directory
```bash
cd /home/ros_master/computer_vision
```

### Step 2: Install Dependencies (if not already installed)
```bash
pip3 install -r requirements.txt
```

---

## Task 1: Custom Object Detection

### Step 1: Setup and Generate Sample Data
```bash
python3 run_task.py --task task1 --mode demo
```

**Expected Output:**
- Creates directory structure
- Generates 20 sample images (16 train, 4 validation)
- Creates COCO-format annotations

**Verify:**
```bash
ls -la task1_object_detection/data/train/images/ | head -5
ls -la task1_object_detection/data/val/images/ | head -5
```

### Step 2: Run Full Training (Optional - takes time)
```bash
cd task1_object_detection
python3 train.py --config configs/default.yaml
```

**Note:** This will train the model. It may take a while depending on your hardware.

**To check training progress:**
```bash
# Check if checkpoint was created
ls -lh checkpoints/
```

### Step 3: Evaluate Model (if training completed)
```bash
python3 evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_model.pth
```

**Expected Output:**
- mAP score
- FPS (inference speed)
- Model size
- Results saved to `results/evaluation_results.txt`

### Step 4: Run Inference
```bash
# Test on validation images
python3 inference.py --model checkpoints/best_model.pth --input data/val/images --output results/
```

**Verify Results:**
```bash
ls -la results/
cat results/evaluation_results.txt 2>/dev/null || echo "Check results directory"
```

### Step 5: Return to Root Directory
```bash
cd ..
```

---

## Task 2: Quality Inspection System

### Step 1: Setup and Generate Sample Data
```bash
python3 run_task.py --task task2 --mode demo
```

**Expected Output:**
- Creates sample defective images (4 types: scratch, misalignment, missing_component, discoloration)
- Creates non-defective images
- Creates annotations in COCO format

**Verify:**
```bash
ls -la task2_quality_inspection/samples/defective/
ls -la task2_quality_inspection/samples/non_defective/
ls -la task2_quality_inspection/samples/annotations/
```

### Step 2: Run Full Training (Optional - takes time)
```bash
cd task2_quality_inspection
python3 train_inspection_model.py --config config.yaml
```

**To check training progress:**
```bash
ls -lh checkpoints/
```

### Step 3: Run Inspection (if training completed, or test with demo)
```bash
# If you have a trained model:
python3 inspect.py --model checkpoints/best_model.pth --input samples/defective --output results/

# Or test on a single image:
python3 inspect.py --model checkpoints/best_model.pth --input samples/defective/scratch_001.jpg --output results/
```

**Expected Output:**
- Annotated images with bounding boxes
- JSON files with defect details including:
  - Defect type and confidence
  - Bounding box coordinates
  - Center coordinates (x, y)
  - Severity assessment

**Verify Results:**
```bash
ls -la results/
cat results/*.json | head -30  # View JSON results
```

### Step 4: Return to Root Directory
```bash
cd ..
```

---

## Task 3: VLM Design Document

### Step 1: View the Design Document
```bash
python3 run_task.py --task task3
```

**Or directly:**
```bash
cat task3_vlm_design/VLM_Design_Document.md
```

**Or view with a pager:**
```bash
less task3_vlm_design/VLM_Design_Document.md
```

### Step 2: Check Document Sections
```bash
grep -E "^## \(" task3_vlm_design/VLM_Design_Document.md
```

**Expected Output:**
- (A) Model Selection
- (B) Design Strategy
- (C) Optimization
- (D) Hallucination Mitigation
- (E) Training Plan
- (F) Validation

### Step 3: Get Document Statistics
```bash
wc -l task3_vlm_design/VLM_Design_Document.md
ls -lh task3_vlm_design/VLM_Design_Document.md
```

---

## Quick Test All Tasks (Demo Mode)

If you want to quickly verify all tasks are set up correctly:

```bash
# Test Task 1
python3 run_task.py --task task1 --mode demo

# Test Task 2
python3 run_task.py --task task2 --mode demo

# Test Task 3
python3 run_task.py --task task3
```

---

## Verification Checklist

After running each task, verify:

### Task 1 Verification:
```bash
# Check data was generated
ls task1_object_detection/data/train/images/*.jpg | wc -l  # Should show 16
ls task1_object_detection/data/val/images/*.jpg | wc -l    # Should show 4

# Check annotations exist
test -f task1_object_detection/data/train/annotations.json && echo "✓ Train annotations exist"
test -f task1_object_detection/data/val/annotations.json && echo "✓ Val annotations exist"

# If trained, check results
test -d task1_object_detection/results && echo "✓ Results directory exists"
test -f task1_object_detection/checkpoints/best_model.pth && echo "✓ Model checkpoint exists"
```

### Task 2 Verification:
```bash
# Check sample images
ls task2_quality_inspection/samples/defective/*.jpg | wc -l  # Should show 4
ls task2_quality_inspection/samples/non_defective/*.jpg | wc -l  # Should show 2

# Check annotations
ls task2_quality_inspection/samples/annotations/*.json | wc -l  # Should show 4

# If trained, check results
test -d task2_quality_inspection/results && echo "✓ Results directory exists"
test -f task2_quality_inspection/checkpoints/best_model.pth && echo "✓ Model checkpoint exists"
```

### Task 3 Verification:
```bash
# Check document exists
test -f task3_vlm_design/VLM_Design_Document.md && echo "✓ Design document exists"

# Check document size (should be ~24KB)
ls -lh task3_vlm_design/VLM_Design_Document.md
```

---

## Troubleshooting

### If you get "python: command not found"
Use `python3` instead of `python`:
```bash
python3 run_task.py --task task1 --mode demo
```

### If training fails due to missing data
Generate data first:
```bash
python3 scripts/generate_sample_data.py --task task1
python3 scripts/generate_sample_data.py --task task2
```

### If you get import errors
Make sure you're in the correct directory:
```bash
pwd  # Should show /home/ros_master/computer_vision
```

### If checkpoint not found
Training may not have completed. Check:
```bash
ls -la task1_object_detection/checkpoints/
ls -la task2_quality_inspection/checkpoints/
```

---

## Expected File Structure After Testing

```
computer_vision/
├── task1_object_detection/
│   ├── data/
│   │   ├── train/
│   │   │   ├── images/          # 16 images
│   │   │   └── annotations.json
│   │   └── val/
│   │       ├── images/          # 4 images
│   │       └── annotations.json
│   ├── checkpoints/             # After training
│   └── results/                 # After evaluation/inference
│
├── task2_quality_inspection/
│   ├── samples/
│   │   ├── defective/           # 4 images
│   │   ├── non_defective/       # 2 images
│   │   └── annotations/          # 4 JSON files
│   ├── checkpoints/             # After training
│   └── results/                 # After inspection
│
└── task3_vlm_design/
    └── VLM_Design_Document.md   # Design document
```

---

## Quick Reference Commands

```bash
# Task 1 - Quick test
python3 run_task.py --task task1 --mode demo

# Task 1 - Full training (takes time)
cd task1_object_detection && python3 train.py --config configs/default.yaml && cd ..

# Task 2 - Quick test
python3 run_task.py --task task2 --mode demo

# Task 2 - Full training (takes time)
cd task2_quality_inspection && python3 train_inspection_model.py --config config.yaml && cd ..

# Task 3 - View document
python3 run_task.py --task task3
```

---

## Notes

- **Demo mode** is fast and just sets up data structures
- **Full mode** actually trains models (can take 30+ minutes depending on hardware)
- Sample data is synthetic - for real results, use actual datasets
- All commands assume you're in `/home/ros_master/computer_vision` directory


