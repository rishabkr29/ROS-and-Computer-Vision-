"""
Evaluation metrics for object detection
"""

import numpy as np
from collections import defaultdict


def compute_iou(box1, box2):
    """Compute IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_ap(predictions, targets, iou_threshold=0.5):
    """Compute Average Precision for a single class"""
    if len(predictions) == 0 and len(targets) == 0:
        return 1.0
    if len(predictions) == 0:
        return 0.0
    if len(targets) == 0:
        return 0.0
    
    # Sort predictions by score
    sorted_indices = np.argsort([-p['score'] for p in predictions])
    sorted_predictions = [predictions[i] for i in sorted_indices]
    
    # Match predictions to targets
    matched = [False] * len(targets)
    tp = []
    fp = []
    
    for pred in sorted_predictions:
        best_iou = 0.0
        best_match = -1
        
        for i, target in enumerate(targets):
            if matched[i]:
                continue
            
            iou = compute_iou(pred['box'], target['box'])
            if iou > best_iou:
                best_iou = iou
                best_match = i
        
        if best_iou >= iou_threshold:
            matched[best_match] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    
    # Compute precision and recall
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    
    recalls = tp / len(targets) if len(targets) > 0 else np.zeros_like(tp)
    precisions = tp / (tp + fp) if (tp + fp).sum() > 0 else np.zeros_like(tp)
    
    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap


def compute_map(predictions, targets, num_classes, iou_threshold=0.5):
    """Compute mean Average Precision"""
    # Group by class
    class_predictions = defaultdict(list)
    class_targets = defaultdict(list)
    
    for pred, target in zip(predictions, targets):
        # Process predictions
        for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
            class_predictions[label].append({
                'box': box,
                'score': score
            })
        
        # Process targets
        for box, label in zip(target['boxes'], target['labels']):
            class_targets[label].append({
                'box': box
            })
    
    # Compute AP for each class
    aps = []
    for cls_id in range(num_classes):
        cls_preds = class_predictions.get(cls_id, [])
        cls_targets = class_targets.get(cls_id, [])
        ap = compute_ap(cls_preds, cls_targets, iou_threshold)
        aps.append(ap)
    
    # Compute mAP
    map_score = np.mean(aps) if len(aps) > 0 else 0.0
    return map_score



