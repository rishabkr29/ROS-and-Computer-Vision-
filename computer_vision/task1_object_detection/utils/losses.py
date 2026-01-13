"""
Loss functions for Faster R-CNN training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_l1_loss(input, target, beta=1.0):
    """Smooth L1 loss"""
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean()


def compute_rpn_loss(rpn_cls_logits, rpn_bbox_deltas, anchors, target, 
                     positive_iou_threshold=0.7, negative_iou_threshold=0.3):
    """Compute RPN loss"""
    device = rpn_cls_logits.device
    
    # Match anchors to ground truth
    gt_boxes = target['boxes']
    gt_labels = target['labels']
    
    if len(gt_boxes) == 0:
        # No ground truth, all negatives
        num_anchors = rpn_cls_logits.size(1)
        labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        matched_gt_boxes = torch.zeros((num_anchors, 4), device=device)
    else:
        # Compute IoU between anchors and GT boxes
        ious = compute_iou(anchors, gt_boxes)
        max_ious, matched_gt_indices = ious.max(dim=1)
        
        # Assign labels
        labels = torch.zeros(len(anchors), dtype=torch.long, device=device)
        labels[max_ious >= positive_iou_threshold] = 1
        labels[max_ious < negative_iou_threshold] = 0
        
        # Get matched GT boxes
        matched_gt_boxes = gt_boxes[matched_gt_indices]
    
    # Classification loss (binary cross entropy)
    rpn_cls_loss = F.binary_cross_entropy_with_logits(
        rpn_cls_logits.squeeze(-1), labels.float(), reduction='mean'
    )
    
    # Regression loss (only for positive anchors)
    positive_mask = labels == 1
    if positive_mask.sum() > 0:
        positive_anchors = anchors[positive_mask]
        positive_gt_boxes = matched_gt_boxes[positive_mask]
        
        # Compute deltas
        anchor_deltas = compute_bbox_deltas(positive_anchors, positive_gt_boxes)
        pred_deltas = rpn_bbox_deltas[positive_mask]
        
        rpn_reg_loss = smooth_l1_loss(pred_deltas, anchor_deltas)
    else:
        rpn_reg_loss = torch.tensor(0.0, device=device)
    
    return rpn_cls_loss, rpn_reg_loss


def compute_roi_loss(roi_cls_logits, roi_bbox_deltas, proposals, target):
    """Compute ROI head loss"""
    device = roi_cls_logits.device
    gt_boxes = target['boxes']
    gt_labels = target['labels']
    
    if len(gt_boxes) == 0 or len(proposals) == 0:
        # No ground truth or proposals
        cls_loss = F.cross_entropy(roi_cls_logits, 
                                   torch.zeros(len(proposals), dtype=torch.long, device=device))
        reg_loss = torch.tensor(0.0, device=device)
        return cls_loss, reg_loss
    
    # Match proposals to ground truth
    ious = compute_iou(proposals, gt_boxes)
    max_ious, matched_gt_indices = ious.max(dim=1)
    
    # Assign labels (background = 0)
    labels = torch.zeros(len(proposals), dtype=torch.long, device=device)
    positive_mask = max_ious >= 0.5
    labels[positive_mask] = gt_labels[matched_gt_indices[positive_mask]] + 1  # +1 for background
    
    # Classification loss
    cls_loss = F.cross_entropy(roi_cls_logits, labels)
    
    # Regression loss (only for positive proposals)
    if positive_mask.sum() > 0:
        positive_proposals = proposals[positive_mask]
        positive_gt_boxes = gt_boxes[matched_gt_indices[positive_mask]]
        
        # Compute deltas
        gt_deltas = compute_bbox_deltas(positive_proposals, positive_gt_boxes)
        pred_deltas = roi_bbox_deltas[positive_mask]
        
        # Only compute loss for correct class
        matched_labels = labels[positive_mask]
        pred_deltas = pred_deltas.view(-1, roi_bbox_deltas.size(1) // 4, 4)
        pred_deltas = pred_deltas[torch.arange(len(pred_deltas)), matched_labels - 1]
        
        reg_loss = smooth_l1_loss(pred_deltas, gt_deltas)
    else:
        reg_loss = torch.tensor(0.0, device=device)
    
    return cls_loss, reg_loss


def compute_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes"""
    # boxes: [N, 4] in format [x1, y1, x2, y2]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Compute intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    # Compute union
    union = area1[:, None] + area2 - inter
    
    iou = inter / union
    return iou


def compute_bbox_deltas(anchors, gt_boxes):
    """Compute bbox deltas from anchors to GT boxes"""
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    cx = anchors[:, 0] + 0.5 * widths
    cy = anchors[:, 1] + 0.5 * heights
    
    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_cx = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_cy = gt_boxes[:, 1] + 0.5 * gt_heights
    
    dx = (gt_cx - cx) / widths
    dy = (gt_cy - cy) / heights
    dw = torch.log(gt_widths / widths)
    dh = torch.log(gt_heights / heights)
    
    deltas = torch.stack([dx, dy, dw, dh], dim=1)
    return deltas

