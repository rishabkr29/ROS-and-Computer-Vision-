#!/usr/bin/env python3
"""
Prepare individual tasks for GitHub upload
Creates clean directories with only necessary files
"""

import os
import shutil
import json
from pathlib import Path
import argparse


def create_gitignore(output_dir):
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Model checkpoints (large files - exclude)
checkpoints/*.pth
*.pth
*.pt

# Large data files (exclude, keep structure)
data/train/images/*.jpg
data/train/images/*.png
data/val/images/*.jpg
data/val/images/*.png
samples/defective/*.jpg
samples/non_defective/*.jpg

# Results (optional - comment out if you want to include)
# results/
# *.txt
# *.json

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log
tensorboard/

# Temporary files
*.tmp
*.bak
"""
    
    with open(output_dir / ".gitignore", "w") as f:
        f.write(gitignore_content)
    print(f"Created .gitignore in {output_dir}")


def prepare_task1(output_dir):
    """Prepare Task 1 for upload"""
    print("Preparing Task 1: Custom Object Detection...")
    
    source = Path("task1_object_detection")
    dest = output_dir / "task1_object_detection"
    
    # Copy entire task directory
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)
    
    # Copy requirements.txt
    shutil.copy("requirements.txt", output_dir / "requirements.txt")
    
    # Create README
    readme_content = """# Task 1: Custom Object Detection with Faster R-CNN

## Overview
This repository contains a complete implementation of Faster R-CNN for object detection, trained from scratch (no pre-trained weights).

## Features
- Custom Faster R-CNN implementation
- Training from scratch
- Evaluation metrics (mAP, FPS, model size)
- Real-time inference support
- Video processing capability

## Installation
```bash
pip install -r requirements.txt
```

## Usage

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
python inference.py --model checkpoints/best_model.pth --input image.jpg --output results/
```

## Dataset Format
Uses COCO annotation format. See `data/train/annotations.json` for example.

## Results
- Evaluation results: `results/evaluation_results.txt`
- Training summary: `results/training_summary.txt`
- Annotated images: `results/`

## Model Architecture
- Backbone: Custom CNN
- RPN: Region Proposal Network
- ROI Head: Classification and regression

## Requirements
- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU training)

## License
[Your License Here]
"""
    
    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    create_gitignore(output_dir)
    
    print(f"Task 1 prepared in: {output_dir}")
    print(f"Files ready for upload: {len(list(dest.rglob('*')))} files")


def prepare_task2(output_dir):
    """Prepare Task 2 for upload"""
    print("Preparing Task 2: Quality Inspection...")
    
    source = Path("task2_quality_inspection")
    dest = output_dir / "task2_quality_inspection"
    
    # Copy entire task directory
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)
    
    # Copy requirements.txt
    shutil.copy("requirements.txt", output_dir / "requirements.txt")
    
    # Create README
    readme_content = """# Task 2: Automated Quality Inspection System

## Overview
Automated visual inspection system for manufacturing defects. Detects and classifies defects with localization and severity assessment.

## Features
- Defect detection (4 types: scratch, misalignment, missing component, discoloration)
- Bounding box localization
- Center coordinate extraction (x, y)
- Severity assessment (High/Medium/Low)
- JSON output with structured defect information

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python train_inspection_model.py --config config.yaml
```

### Inspection
```bash
# Single image
python inspect.py --model checkpoints/best_model.pth --input sample.jpg --output results/

# Directory of images
python inspect.py --model checkpoints/best_model.pth --input images/ --output results/
```

## Output Format
JSON files with:
- Defect type and confidence score
- Bounding box coordinates
- Center coordinates (x, y)
- Severity assessment

## Defect Types
1. **Scratch**: Surface scratches or abrasions
2. **Misalignment**: Components not properly aligned
3. **Missing Component**: Missing parts or features
4. **Discoloration**: Color variations or stains

## Sample Data
Sample images and annotations provided in `samples/` directory.

## Requirements
- Python 3.8+
- PyTorch 1.12+
- OpenCV
- Albumentations

## License
[Your License Here]
"""
    
    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    create_gitignore(output_dir)
    
    print(f"Task 2 prepared in: {output_dir}")
    print(f"Files ready for upload: {len(list(dest.rglob('*')))} files")


def prepare_task3(output_dir):
    """Prepare Task 3 for upload"""
    print("Preparing Task 3: VLM Design Document...")
    
    source = Path("task3_vlm_design")
    dest = output_dir / "task3_vlm_design"
    
    # Copy entire task directory
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)
    
    # Create README
    readme_content = """# Task 3: Custom VLM Design for Industrial Quality Inspection

## Overview
Comprehensive design document for a custom Vision-Language Model (VLM) tailored for offline PCB inspection in semiconductor manufacturing.

## Document Sections
- **(A) Model Selection**: Qwen-VL based architecture with rationale
- **(B) Design Strategy**: Architecture modifications for PCB-specific requirements
- **(C) Optimization**: Techniques for <2s inference and offline deployment
- **(D) Hallucination Mitigation**: Strategies to reduce false information
- **(E) Training Plan**: Multi-stage training approach with QA pair generation
- **(F) Validation**: Methodology for counting accuracy, localization precision, and hallucination rates

## Key Features
- Model selection: Qwen-VL based custom architecture
- Inference time: <2s target
- Offline deployment: <2GB model size, <8GB memory
- Hallucination rate: <3% target
- Localization accuracy: >90% IoU@0.5

## Document
See `VLM_Design_Document.md` for the complete design document.

## Requirements
- Markdown viewer (GitHub, VS Code, etc.)
- No dependencies required (documentation only)

## License
[Your License Here]
"""
    
    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    create_gitignore(output_dir)
    
    print(f"Task 3 prepared in: {output_dir}")
    print(f"Files ready for upload: {len(list(dest.rglob('*')))} files")


def main():
    parser = argparse.ArgumentParser(description='Prepare tasks for GitHub upload')
    parser.add_argument('--task', type=str, required=True,
                       choices=['task1', 'task2', 'task3', 'all'],
                       help='Which task to prepare')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: task1_github, task2_github, etc.)')
    args = parser.parse_args()
    
    base_dir = Path.cwd()
    
    if args.task == 'all':
        tasks = ['task1', 'task2', 'task3']
    else:
        tasks = [args.task]
    
    for task in tasks:
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = base_dir / f"{task}_github"
        
        output_dir.mkdir(exist_ok=True)
        
        if task == 'task1':
            prepare_task1(output_dir)
        elif task == 'task2':
            prepare_task2(output_dir)
        elif task == 'task3':
            prepare_task3(output_dir)
        
        print(f"\n✓ {task.upper()} ready in: {output_dir}")
        print(f"  Next steps:")
        print(f"  1. cd {output_dir}")
        print(f"  2. git init")
        print(f"  3. git add .")
        print(f"  4. git commit -m 'Initial commit: {task}'")
        print(f"  5. Create GitHub repo and push")
        print()


if __name__ == "__main__":
    main()


