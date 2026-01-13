#!/usr/bin/env python3
"""
Generate sample synthetic data for testing tasks
"""

import numpy as np
import cv2
import json
import os
import sys
from pathlib import Path
import argparse
import random
import shutil


def generate_task1_data(output_dir=None, num_images=20):
    """Generate sample images and annotations for Task 1"""
    if output_dir is None:
        # Get script directory and find project root
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        output_dir = project_root / "task1_object_detection" / "data"
    else:
        output_dir = Path(output_dir)
    
    print(f"Generating {num_images} sample images for Task 1...")
    print(f"Output directory: {output_dir}")
    
    base_dir = Path(output_dir)
    train_dir = base_dir / "train" / "images"
    val_dir = base_dir / "val" / "images"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Class names
    classes = ["person", "car", "bicycle", "dog", "cat"]
    num_classes = len(classes)
    
    # Split images
    num_train = int(num_images * 0.8)
    num_val = num_images - num_train
    
    train_images = []
    train_annotations = []
    val_images = []
    val_annotations = []
    
    annotation_id = 1
    
    # Generate train images
    for i in range(num_train):
        # Create random image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some random shapes as "objects"
        num_objects = random.randint(1, 3)
        boxes = []
        labels = []
        
        for _ in range(num_objects):
            x1 = random.randint(50, 500)
            y1 = random.randint(50, 400)
            w = random.randint(50, 150)
            h = random.randint(50, 150)
            x2 = min(x1 + w, 639)
            y2 = min(y1 + h, 479)
            
            # Draw rectangle
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            
            boxes.append([x1, y1, x2, y2])
            labels.append(random.randint(0, num_classes - 1))
        
        # Save image
        filename = f"train_{i+1:03d}.jpg"
        filepath = train_dir / filename
        cv2.imwrite(str(filepath), img)
        
        # Create annotation
        train_images.append({
            "id": i + 1,
            "file_name": filename,
            "width": 640,
            "height": 480
        })
        
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            train_annotations.append({
                "id": annotation_id,
                "image_id": i + 1,
                "category_id": label + 1,  # 1-indexed
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO format: [x, y, width, height]
                "area": (x2 - x1) * (y2 - y1)
            })
            annotation_id += 1
    
    # Generate val images
    for i in range(num_val):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        num_objects = random.randint(1, 2)
        boxes = []
        labels = []
        
        for _ in range(num_objects):
            x1 = random.randint(50, 500)
            y1 = random.randint(50, 400)
            w = random.randint(50, 150)
            h = random.randint(50, 150)
            x2 = min(x1 + w, 639)
            y2 = min(y1 + h, 479)
            
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            
            boxes.append([x1, y1, x2, y2])
            labels.append(random.randint(0, num_classes - 1))
        
        filename = f"val_{i+1:03d}.jpg"
        filepath = val_dir / filename
        cv2.imwrite(str(filepath), img)
        
        val_images.append({
            "id": i + 1,
            "file_name": filename,
            "width": 640,
            "height": 480
        })
        
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            val_annotations.append({
                "id": annotation_id,
                "image_id": i + 1,
                "category_id": label + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1)
            })
            annotation_id += 1
    
    # Save annotations
    categories = [{"id": i+1, "name": name} for i, name in enumerate(classes)]
    
    train_ann = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories
    }
    
    val_ann = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": categories
    }
    
    with open(base_dir / "train" / "annotations.json", "w") as f:
        json.dump(train_ann, f, indent=2)
    
    with open(base_dir / "val" / "annotations.json", "w") as f:
        json.dump(val_ann, f, indent=2)
    
    print(f"Generated {num_train} train and {num_val} val images")
    print(f"Annotations saved to {base_dir}/train/annotations.json and {base_dir}/val/annotations.json")


def generate_task2_data(output_dir=None, num_images=10):
    """Generate sample images and annotations for Task 2"""
    if output_dir is None:
        # Get script directory and find project root
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        output_dir = project_root / "task2_quality_inspection"
    else:
        output_dir = Path(output_dir)
    
    print(f"Generating {num_images} sample images for Task 2...")
    print(f"Output directory: {output_dir}")
    
    base_dir = Path(output_dir)
    samples_dir = base_dir / "samples"
    defective_dir = samples_dir / "defective"
    non_defective_dir = samples_dir / "non_defective"
    annotations_dir = samples_dir / "annotations"
    
    defective_dir.mkdir(parents=True, exist_ok=True)
    non_defective_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Defect types
    defect_types = {
        1: "scratch",
        2: "misalignment",
        3: "missing_component",
        4: "discoloration"
    }
    
    # Generate defective images
    for i, (defect_id, defect_name) in enumerate(defect_types.items()):
        # Create base image (simulating a PCB/manufactured item)
        img = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray background
        
        # Add some base features
        cv2.rectangle(img, (100, 100), (300, 200), (100, 100, 100), -1)  # Component
        cv2.rectangle(img, (350, 150), (550, 250), (100, 100, 100), -1)  # Component
        
        # Add defect based on type
        if defect_name == "scratch":
            # Draw a scratch line
            cv2.line(img, (150, 120), (250, 180), (50, 50, 50), 3)
        elif defect_name == "misalignment":
            # Misaligned component
            cv2.rectangle(img, (120, 120), (320, 220), (150, 50, 50), -1)
        elif defect_name == "missing_component":
            # Missing component (just background)
            pass
        elif defect_name == "discoloration":
            # Discolored area
            cv2.circle(img, (200, 150), 40, (200, 100, 100), -1)
        
        # Save image
        filename = f"{defect_name}_{i+1:03d}.jpg"
        filepath = defective_dir / filename
        cv2.imwrite(str(filepath), img)
        
        # Create annotation
        # Calculate bounding box for defect
        if defect_name == "scratch":
            bbox = [150, 120, 100, 60]  # [x, y, width, height]
        elif defect_name == "misalignment":
            bbox = [120, 120, 200, 100]
        elif defect_name == "missing_component":
            bbox = [400, 200, 100, 80]  # Where component should be
        else:  # discoloration
            bbox = [160, 110, 80, 80]
        
        annotation = {
            "images": [{
                "id": i + 1,
                "file_name": filename,
                "width": 640,
                "height": 480
            }],
            "annotations": [{
                "id": 1,
                "image_id": i + 1,
                "category_id": defect_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3]
            }],
            "categories": [
                {"id": 1, "name": "scratch"},
                {"id": 2, "name": "misalignment"},
                {"id": 3, "name": "missing_component"},
                {"id": 4, "name": "discoloration"}
            ]
        }
        
        ann_file = annotations_dir / f"{defect_name}_{i+1:03d}.json"
        with open(ann_file, "w") as f:
            json.dump(annotation, f, indent=2)
    
    # Generate non-defective images
    for i in range(2):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 200
        cv2.rectangle(img, (100, 100), (300, 200), (100, 100, 100), -1)
        cv2.rectangle(img, (350, 150), (550, 250), (100, 100, 100), -1)
        
        filename = f"good_{i+1:03d}.jpg"
        filepath = non_defective_dir / filename
        cv2.imwrite(str(filepath), img)
    
    # Also create training data structure
    train_dir = base_dir / "data" / "train" / "images"
    val_dir = base_dir / "data" / "val" / "images"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy some images to train/val
    all_defective = list(defective_dir.glob("*.jpg"))
    for img_file in all_defective[:3]:
        shutil.copy(img_file, train_dir / img_file.name)
    for img_file in all_defective[3:]:
        shutil.copy(img_file, val_dir / img_file.name)
    
    # Create combined annotations for training
    train_images = []
    train_annotations = []
    ann_id = 1
    
    for i, ann_file in enumerate(list(annotations_dir.glob("*.json"))[:3]):
        with open(ann_file, "r") as f:
            ann_data = json.load(f)
            train_images.extend(ann_data["images"])
            for ann in ann_data["annotations"]:
                ann["id"] = ann_id
                train_annotations.append(ann)
                ann_id += 1
    
    train_ann = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": [
            {"id": 1, "name": "scratch"},
            {"id": 2, "name": "misalignment"},
            {"id": 3, "name": "missing_component"},
            {"id": 4, "name": "discoloration"}
        ]
    }
    
    with open(base_dir / "data" / "train" / "annotations.json", "w") as f:
        json.dump(train_ann, f, indent=2)
    
    # Val annotations
    val_images = []
    val_annotations = []
    
    for i, ann_file in enumerate(list(annotations_dir.glob("*.json"))[3:]):
        with open(ann_file, "r") as f:
            ann_data = json.load(f)
            val_images.extend(ann_data["images"])
            for ann in ann_data["annotations"]:
                ann["id"] = ann_id
                val_annotations.append(ann)
                ann_id += 1
    
    val_ann = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": train_ann["categories"]
    }
    
    with open(base_dir / "data" / "val" / "annotations.json", "w") as f:
        json.dump(val_ann, f, indent=2)
    
    print(f"Generated {len(defect_types)} defective and 2 non-defective sample images")
    print(f"Annotations saved to {samples_dir}/annotations/")


def main():
    parser = argparse.ArgumentParser(description='Generate sample data for tasks')
    parser.add_argument('--task', type=str, required=True,
                       choices=['task1', 'task2', 'all'],
                       help='Which task to generate data for')
    parser.add_argument('--num-images', type=int, default=20,
                       help='Number of images to generate (for task1)')
    args = parser.parse_args()
    
    if args.task in ['task1', 'all']:
        generate_task1_data(num_images=args.num_images)
    
    if args.task in ['task2', 'all']:
        generate_task2_data()


if __name__ == "__main__":
    main()

