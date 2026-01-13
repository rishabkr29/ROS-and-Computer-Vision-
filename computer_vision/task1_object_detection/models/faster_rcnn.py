"""
Faster R-CNN Implementation from Scratch
Custom object detection model without pre-trained weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


class AnchorGenerator:
    """Generate anchor boxes for RPN"""
    
    def __init__(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        
    def generate_anchors(self, feature_map_size, stride=16):
        """Generate anchors for a feature map"""
        h, w = feature_map_size
        anchors = []
        
        for size in self.sizes:
            for aspect_ratio in self.aspect_ratios:
                base_w = size * (aspect_ratio ** 0.5)
                base_h = size / (aspect_ratio ** 0.5)
                
                for y in range(h):
                    for x in range(w):
                        cx = (x + 0.5) * stride
                        cy = (y + 0.5) * stride
                        anchors.append([cx - base_w/2, cy - base_h/2, 
                                       cx + base_w/2, cy + base_h/2])
        
        return torch.tensor(anchors, dtype=torch.float32)


class Backbone(nn.Module):
    """Custom CNN backbone for feature extraction"""
    
    def __init__(self, in_channels=3, out_channels=256):
        super(Backbone, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Reduce to output channels
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class RegionProposalNetwork(nn.Module):
    """RPN for generating region proposals"""
    
    def __init__(self, in_channels=256, num_anchors=15):
        super(RegionProposalNetwork, self).__init__()
        self.num_anchors = num_anchors
        
        # Shared convolutional layer
        self.conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        
        # Classification head (objectness score)
        self.cls_head = nn.Conv2d(256, num_anchors, kernel_size=1)
        
        # Regression head (bbox refinement)
        self.reg_head = nn.Conv2d(256, num_anchors * 4, kernel_size=1)
        
    def forward(self, features):
        x = F.relu(self.conv(features))
        
        # Objectness scores
        cls_logits = self.cls_head(x)
        cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous()
        cls_logits = cls_logits.view(cls_logits.size(0), -1, 1)
        
        # Bounding box deltas
        bbox_deltas = self.reg_head(x)
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = bbox_deltas.view(bbox_deltas.size(0), -1, 4)
        
        return cls_logits, bbox_deltas


class ROIHead(nn.Module):
    """ROI Head for classification and bbox regression"""
    
    def __init__(self, in_channels=256, num_classes=5, roi_size=7):
        super(ROIHead, self).__init__()
        self.roi_size = roi_size
        
        # ROI pooling equivalent (simplified)
        self.roi_pool = nn.AdaptiveAvgPool2d((roi_size, roi_size))
        
        # Classification and regression heads
        self.fc1 = nn.Linear(in_channels * roi_size * roi_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        
        self.cls_head = nn.Linear(1024, num_classes)
        self.reg_head = nn.Linear(1024, num_classes * 4)
        
    def forward(self, features, rois):
        # Extract ROI features (simplified - in practice use ROIAlign)
        batch_size = features.size(0)
        num_rois = rois.size(0)
        
        # For simplicity, we'll use global average pooling
        # In production, use proper ROIAlign
        pooled_features = self.roi_pool(features)
        pooled_features = pooled_features.view(batch_size, -1)
        
        # Expand for each ROI
        if num_rois > 0:
            pooled_features = pooled_features.unsqueeze(0).expand(num_rois, -1)
        else:
            pooled_features = pooled_features.unsqueeze(0)
        
        x = F.relu(self.fc1(pooled_features))
        x = F.relu(self.fc2(x))
        
        cls_logits = self.cls_head(x)
        bbox_deltas = self.reg_head(x)
        
        return cls_logits, bbox_deltas


class FasterRCNN(nn.Module):
    """Complete Faster R-CNN model"""
    
    def __init__(self, num_classes=5, in_channels=3):
        super(FasterRCNN, self).__init__()
        self.num_classes = num_classes
        
        # Backbone
        self.backbone = Backbone(in_channels=in_channels, out_channels=256)
        
        # RPN
        self.rpn = RegionProposalNetwork(in_channels=256, num_anchors=15)
        
        # ROI Head
        self.roi_head = ROIHead(in_channels=256, num_classes=num_classes)
        
        # Anchor generator
        self.anchor_generator = AnchorGenerator()
        
    def forward(self, images, targets=None):
        # Extract features
        features = self.backbone(images)
        
        # Get feature map size
        _, _, h, w = features.shape
        
        # Generate anchors
        anchors = self.anchor_generator.generate_anchors((h, w))
        anchors = anchors.to(images.device)
        
        # RPN forward
        rpn_cls_logits, rpn_bbox_deltas = self.rpn(features)
        
        if self.training:
            # Training mode
            return {
                'rpn_cls_logits': rpn_cls_logits,
                'rpn_bbox_deltas': rpn_bbox_deltas,
                'anchors': anchors,
                'features': features
            }
        else:
            # Inference mode
            # Apply RPN to get proposals
            proposals = self._generate_proposals(
                anchors, rpn_cls_logits, rpn_bbox_deltas, 
                images.shape[-2:]
            )
            
            # Apply ROI head
            roi_cls_logits, roi_bbox_deltas = self.roi_head(features, proposals)
            
            # Post-process
            boxes, scores, labels = self._post_process(
                proposals, roi_cls_logits, roi_bbox_deltas
            )
            
            return boxes, scores, labels
    
    def _generate_proposals(self, anchors, cls_logits, bbox_deltas, image_size):
        """Generate proposals from RPN output"""
        # Apply sigmoid to get objectness scores
        objectness = torch.sigmoid(cls_logits.squeeze(-1))
        
        # Filter by threshold
        keep = objectness > 0.7
        if keep.sum() == 0:
            return torch.empty((0, 4), device=anchors.device)
        
        # Apply bbox deltas
        proposals = self._apply_deltas(anchors[keep], bbox_deltas[keep])
        
        # Clip to image boundaries
        proposals[:, [0, 2]] = proposals[:, [0, 2]].clamp(0, image_size[1])
        proposals[:, [1, 3]] = proposals[:, [1, 3]].clamp(0, image_size[0])
        
        # NMS
        keep = nms(proposals, objectness[keep], iou_threshold=0.7)
        proposals = proposals[keep[:100]]  # Top 100 proposals
        
        return proposals
    
    def _apply_deltas(self, anchors, deltas):
        """Apply bbox deltas to anchors"""
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        cx = anchors[:, 0] + 0.5 * widths
        cy = anchors[:, 1] + 0.5 * heights
        
        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]
        
        pred_cx = cx + dx * widths
        pred_cy = cy + dy * heights
        pred_w = widths * torch.exp(dw)
        pred_h = heights * torch.exp(dh)
        
        boxes = torch.zeros_like(anchors)
        boxes[:, 0] = pred_cx - 0.5 * pred_w
        boxes[:, 1] = pred_cy - 0.5 * pred_h
        boxes[:, 2] = pred_cx + 0.5 * pred_w
        boxes[:, 3] = pred_cy + 0.5 * pred_h
        
        return boxes
    
    def _post_process(self, proposals, cls_logits, bbox_deltas):
        """Post-process ROI head output"""
        # Apply softmax to get class probabilities
        cls_probs = F.softmax(cls_logits, dim=-1)
        
        # Get max scores and labels
        scores, labels = cls_probs.max(dim=-1)
        
        # Filter by score threshold
        keep = scores > 0.5
        if keep.sum() == 0:
            return [], [], []
        
        proposals = proposals[keep]
        scores = scores[keep]
        labels = labels[keep]
        bbox_deltas = bbox_deltas[keep]
        
        # Apply bbox deltas
        boxes = self._apply_deltas(proposals, bbox_deltas)
        
        # NMS per class
        final_boxes = []
        final_scores = []
        final_labels = []
        
        for cls_id in range(1, self.num_classes):  # Skip background
            cls_mask = labels == cls_id
            if cls_mask.sum() == 0:
                continue
            
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            
            keep = nms(cls_boxes, cls_scores, iou_threshold=0.5)
            final_boxes.append(cls_boxes[keep])
            final_scores.append(cls_scores[keep])
            final_labels.append(torch.full((len(keep),), cls_id, device=boxes.device))
        
        if len(final_boxes) == 0:
            return [], [], []
        
        return torch.cat(final_boxes), torch.cat(final_scores), torch.cat(final_labels)



