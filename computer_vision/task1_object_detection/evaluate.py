"""
Evaluation script for Faster R-CNN
Computes mAP, FPS, and model size
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import yaml
import os
import time
import numpy as np
from tqdm import tqdm

from models.faster_rcnn import FasterRCNN
from utils.dataset import ObjectDetectionDataset, collate_fn
from utils.metrics import compute_map


def evaluate_model(model, dataloader, device, num_classes):
    """Evaluate model and compute metrics"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    inference_times = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            
            # Measure inference time
            start_time = time.time()
            boxes, scores, labels = model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time / len(images))  # Per image
            
            # Store predictions and targets
            for i, target in enumerate(targets):
                if isinstance(boxes, list) or len(boxes) == 0:
                    pred_boxes = []
                    pred_scores = []
                    pred_labels = []
                else:
                    # Filter predictions for this image
                    pred_boxes = boxes[i].cpu().numpy() if len(boxes) > i else []
                    pred_scores = scores[i].cpu().numpy() if len(scores) > i else []
                    pred_labels = labels[i].cpu().numpy() if len(labels) > i else []
                
                all_predictions.append({
                    'boxes': pred_boxes,
                    'scores': pred_scores,
                    'labels': pred_labels
                })
                
                all_targets.append({
                    'boxes': target['boxes'].cpu().numpy(),
                    'labels': target['labels'].cpu().numpy()
                })
    
    # Compute mAP
    map_score = compute_map(all_predictions, all_targets, num_classes)
    
    # Compute FPS
    avg_inference_time = np.mean(inference_times)
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    return {
        'map': map_score,
        'fps': fps,
        'avg_inference_time': avg_inference_time
    }


def get_model_size(model):
    """Compute model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def main():
    parser = argparse.ArgumentParser(description='Evaluate Faster R-CNN')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataset
    val_dataset = ObjectDetectionDataset(
        image_dir=config['val_image_dir'],
        annotations_file=config['val_annotations'],
        is_training=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    
    # Create model
    num_classes = config.get('num_classes', 5)
    model = FasterRCNN(num_classes=num_classes).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
    
    # Compute model size
    model_size = get_model_size(model)
    print(f'\nModel size: {model_size:.2f} MB')
    
    # Evaluate
    print('\nEvaluating model...')
    metrics = evaluate_model(model, val_loader, device, num_classes)
    
    # Print results
    print('\n' + '='*50)
    print('EVALUATION RESULTS')
    print('='*50)
    print(f'mAP: {metrics["map"]:.4f}')
    print(f'FPS: {metrics["fps"]:.2f}')
    print(f'Average inference time: {metrics["avg_inference_time"]*1000:.2f} ms')
    print(f'Model size: {model_size:.2f} MB')
    print('='*50)
    
    # Save results
    results_dir = config.get('results_dir', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write('Evaluation Results\n')
        f.write('='*50 + '\n')
        f.write(f'mAP: {metrics["map"]:.4f}\n')
        f.write(f'FPS: {metrics["fps"]:.2f}\n')
        f.write(f'Average inference time: {metrics["avg_inference_time"]*1000:.2f} ms\n')
        f.write(f'Model size: {model_size:.2f} MB\n')
    
    print(f'\nResults saved to {results_file}')


if __name__ == '__main__':
    main()



