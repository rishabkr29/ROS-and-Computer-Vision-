"""
Script to download and prepare a dataset for training
This script creates a sample dataset structure
"""

import os
import json
import shutil
from pathlib import Path
import urllib.request
import zipfile


def create_sample_dataset_structure():
    """Create directory structure for dataset"""
    base_dir = Path("data")
    
    # Create directories
    (base_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (base_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    
    print("Created dataset directory structure")
    print("\nTo use this code:")
    print("1. Download PASCAL VOC dataset or create your own dataset")
    print("2. Place images in data/train/images and data/val/images")
    print("3. Create annotations.json files in COCO format")
    print("\nExample annotations.json format:")
    print("""
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
    {"id": 1, "name": "class1"},
    {"id": 2, "name": "class2"}
  ]
}
    """)


def create_dummy_annotations():
    """Create dummy annotation files for testing"""
    # Train annotations
    train_annotations = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "object1"},
            {"id": 2, "name": "object2"},
            {"id": 3, "name": "object3"},
            {"id": 4, "name": "object4"},
            {"id": 5, "name": "object5"}
        ]
    }
    
    # Val annotations
    val_annotations = {
        "images": [],
        "annotations": [],
        "categories": train_annotations["categories"]
    }
    
    # Save
    with open("data/train/annotations.json", "w") as f:
        json.dump(train_annotations, f, indent=2)
    
    with open("data/val/annotations.json", "w") as f:
        json.dump(val_annotations, f, indent=2)
    
    print("Created dummy annotation files")
    print("Note: You need to add actual images and update annotations")


if __name__ == "__main__":
    create_sample_dataset_structure()
    create_dummy_annotations()
    print("\nDataset structure created!")
    print("Please add your images and update the annotation files.")



