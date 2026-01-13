"""
Inference script for real-time object detection
"""

import torch
import cv2
import numpy as np
import argparse
import yaml
import os
from pathlib import Path
import time

from models.faster_rcnn import FasterRCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_model(checkpoint_path, num_classes, device):
    """Load trained model"""
    model = FasterRCNN(num_classes=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def preprocess_image(image, device):
    """Preprocess image for inference"""
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    return image_tensor


def draw_detections(image, boxes, scores, labels, class_names):
    """Draw bounding boxes on image"""
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]
    
    for box, score, label in zip(boxes, scores, labels):
        if score < 0.5:  # Confidence threshold
            continue
        
        x1, y1, x2, y2 = map(int, box)
        color = colors[label % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_text = f'{class_names.get(label, f"Class {label}")}: {score:.2f}'
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(image, (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), color, -1)
        cv2.putText(image, label_text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


def process_image(model, image_path, class_names, device):
    """Process a single image"""
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    
    # Preprocess
    image_tensor = preprocess_image(image_rgb, device)
    
    # Inference
    start_time = time.time()
    with torch.no_grad():
        boxes, scores, labels = model(image_tensor)
    inference_time = time.time() - start_time
    
    # Convert to numpy
    if len(boxes) > 0:
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
    else:
        boxes = np.array([])
        scores = np.array([])
        labels = np.array([])
    
    # Draw detections
    result_image = draw_detections(original_image, boxes, scores, labels, class_names)
    
    # Add FPS info
    fps = 1.0 / inference_time if inference_time > 0 else 0
    cv2.putText(result_image, f'FPS: {fps:.1f}', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return result_image, inference_time, len(boxes)


def process_video(model, video_path, output_path, class_names, device):
    """Process video file"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_time = 0.0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = preprocess_image(image_rgb, device)
        
        start_time = time.time()
        with torch.no_grad():
            boxes, scores, labels = model(image_tensor)
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # Convert to numpy
        if len(boxes) > 0:
            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
        else:
            boxes = np.array([])
            scores = np.array([])
            labels = np.array([])
        
        # Draw detections
        result_frame = draw_detections(frame, boxes, scores, labels, class_names)
        
        # Add FPS info
        current_fps = 1.0 / inference_time if inference_time > 0 else 0
        cv2.putText(result_frame, f'FPS: {current_fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out.write(result_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f'Processed {frame_count} frames, avg FPS: {frame_count/total_time:.2f}')
    
    cap.release()
    out.release()
    print(f'Video processing completed. Average FPS: {frame_count/total_time:.2f}')


def main():
    parser = argparse.ArgumentParser(description='Run inference with Faster R-CNN')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image, video, or directory')
    parser.add_argument('--output', type=str, default='output',
                       help='Path to output directory or file')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Class names
    class_names = config.get('class_names', {
        0: 'Class 0',
        1: 'Class 1',
        2: 'Class 2',
        3: 'Class 3',
        4: 'Class 4'
    })
    
    # Load model
    num_classes = config.get('num_classes', 5)
    print(f'Loading model from {args.model}...')
    model = load_model(args.model, num_classes, device)
    print('Model loaded successfully!')
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Image
            print(f'Processing image: {input_path}')
            result_image, inference_time, num_detections = process_image(
                model, str(input_path), class_names, device
            )
            
            output_path = Path(args.output)
            if output_path.suffix == '':
                output_path = output_path / input_path.name
            
            cv2.imwrite(str(output_path), result_image)
            print(f'Result saved to {output_path}')
            print(f'Inference time: {inference_time*1000:.2f} ms')
            print(f'Detections: {num_detections}')
        
        elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            # Video
            print(f'Processing video: {input_path}')
            process_video(model, str(input_path), args.output, class_names, device)
            print(f'Result saved to {args.output}')
    
    elif input_path.is_dir():
        # Directory of images
        os.makedirs(args.output, exist_ok=True)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f'Processing {len(image_files)} images...')
        total_time = 0.0
        
        for img_file in image_files:
            result_image, inference_time, num_detections = process_image(
                model, str(img_file), class_names, device
            )
            
            output_path = Path(args.output) / img_file.name
            cv2.imwrite(str(output_path), result_image)
            total_time += inference_time
        
        avg_fps = len(image_files) / total_time if total_time > 0 else 0
        print(f'\nProcessed {len(image_files)} images')
        print(f'Average FPS: {avg_fps:.2f}')
        print(f'Results saved to {args.output}')
    
    else:
        print(f'Error: Input path {args.input} does not exist')


if __name__ == '__main__':
    main()



