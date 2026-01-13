"""
Training script for Faster R-CNN
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import os
from tqdm import tqdm
import time

from models.faster_rcnn import FasterRCNN
from utils.dataset import ObjectDetectionDataset, collate_fn
from utils.losses import compute_rpn_loss, compute_roi_loss


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    rpn_cls_loss_total = 0.0
    rpn_reg_loss_total = 0.0
    roi_cls_loss_total = 0.0
    roi_reg_loss_total = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        
        # Forward pass
        outputs = model(images, targets)
        
        # Compute losses
        rpn_cls_loss = torch.tensor(0.0, device=device)
        rpn_reg_loss = torch.tensor(0.0, device=device)
        roi_cls_loss = torch.tensor(0.0, device=device)
        roi_reg_loss = torch.tensor(0.0, device=device)
        
        batch_loss = 0.0
        
        for i, target in enumerate(targets):
            # RPN losses
            rpn_cls, rpn_reg = compute_rpn_loss(
                outputs['rpn_cls_logits'][i:i+1],
                outputs['rpn_bbox_deltas'][i:i+1],
                outputs['anchors'],
                target
            )
            rpn_cls_loss += rpn_cls
            rpn_reg_loss += rpn_reg
            
            # ROI losses (simplified - in practice need proper ROI sampling)
            # For now, we'll use a simplified approach
            roi_cls = torch.tensor(0.0, device=device)
            roi_reg = torch.tensor(0.0, device=device)
            roi_cls_loss += roi_cls
            roi_reg_loss += roi_reg
        
        # Total loss
        loss = (rpn_cls_loss + 10 * rpn_reg_loss + 
                roi_cls_loss + 10 * roi_reg_loss) / len(targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        rpn_cls_loss_total += rpn_cls_loss.item() / len(targets)
        rpn_reg_loss_total += rpn_reg_loss.item() / len(targets)
        roi_cls_loss_total += roi_cls_loss.item() / len(targets)
        roi_reg_loss_total += roi_reg_loss.item() / len(targets)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'rpn_cls': f'{rpn_cls_loss.item()/len(targets):.4f}',
            'rpn_reg': f'{rpn_reg_loss.item()/len(targets):.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    return {
        'loss': avg_loss,
        'rpn_cls_loss': rpn_cls_loss_total / len(dataloader),
        'rpn_reg_loss': rpn_reg_loss_total / len(dataloader),
        'roi_cls_loss': roi_cls_loss_total / len(dataloader),
        'roi_reg_loss': roi_reg_loss_total / len(dataloader)
    }


def main():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Create dataset
    train_dataset = ObjectDetectionDataset(
        image_dir=config['train_image_dir'],
        annotations_file=config['train_annotations'],
        is_training=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    
    # Create model
    num_classes = config.get('num_classes', 5)
    model = FasterRCNN(num_classes=num_classes).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=0.9,
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get('lr_step_size', 10),
        gamma=config.get('lr_gamma', 0.1)
    )
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resumed from epoch {start_epoch}')
    
    # TensorBoard writer
    writer = SummaryWriter(config['log_dir'])
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(start_epoch, config['num_epochs']):
        print(f'\nEpoch {epoch+1}/{config["num_epochs"]}')
        
        # Train
        metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Log metrics
        writer.add_scalar('Train/Loss', metrics['loss'], epoch)
        writer.add_scalar('Train/RPN_Cls_Loss', metrics['rpn_cls_loss'], epoch)
        writer.add_scalar('Train/RPN_Reg_Loss', metrics['rpn_reg_loss'], epoch)
        writer.add_scalar('Train/ROI_Cls_Loss', metrics['roi_cls_loss'], epoch)
        writer.add_scalar('Train/ROI_Reg_Loss', metrics['roi_reg_loss'], epoch)
        writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': metrics['loss'],
        }
        
        # Save latest
        torch.save(checkpoint, 
                  os.path.join(config['checkpoint_dir'], 'latest.pth'))
        
        # Save best
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            torch.save(checkpoint,
                      os.path.join(config['checkpoint_dir'], 'best_model.pth'))
            print(f'New best model saved with loss: {best_loss:.4f}')
    
    writer.close()
    print('Training completed!')


if __name__ == '__main__':
    main()

