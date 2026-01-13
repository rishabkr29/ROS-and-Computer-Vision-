# Custom VLM Design for Industrial Quality Inspection

## Executive Summary

This document presents a comprehensive design for a custom Vision-Language Model (VLM) tailored for offline PCB inspection in semiconductor manufacturing. The system addresses the challenge of natural language queries about defects with structured responses including locations and confidence scores, while maintaining <2s inference time and minimizing hallucinations.

---

## (A) Model Selection

### Recommended Choice: **Custom Architecture Based on Qwen-VL**

### Rationale

After evaluating LLaVA, BLIP-2, Qwen-VL, and custom architectures, **Qwen-VL** serves as the best foundation with significant customizations:

#### Comparison Matrix

| Model | Size | Inference Speed | Fine-tuning Flexibility | Licensing | Architecture Suitability |
|-------|------|----------------|------------------------|-----------|------------------------|
| **LLaVA** | 7B-13B | Medium | High | Apache 2.0 | Good, but slower |
| **BLIP-2** | 3.7B-11B | Fast | Medium | BSD-3 | Limited localization precision |
| **Qwen-VL** | 2B-7B | Fast | High | Apache 2.0 | Excellent for localization |
| **Custom** | Variable | Optimized | Full | Custom | Best for specific use case |

### Why Qwen-VL?

1. **Inference Speed**: Qwen-VL-2B achieves <1s inference on modern GPUs, meeting <2s requirement
2. **Localization Capability**: Built-in spatial understanding with bounding box tokens
3. **Fine-tuning Flexibility**: Supports LoRA, QLoRA, and full fine-tuning
4. **Open Licensing**: Apache 2.0 allows commercial use
5. **Efficient Architecture**: Optimized attention mechanisms for faster inference

### Architectural Modifications for Precise Localization

1. **Spatial Token Enhancement**
   - Add explicit coordinate tokens (x, y, width, height) to vocabulary
   - Implement coordinate-aware attention mechanisms
   - Fine-grained grid-based feature extraction (e.g., 32x32 patches)

2. **Multi-scale Feature Fusion**
   - Combine features from multiple resolution levels
   - Preserve spatial information through skip connections
   - Use Feature Pyramid Network (FPN) in vision encoder

3. **Coordinate Regression Head**
   - Separate head for bounding box regression
   - Direct coordinate prediction alongside text generation
   - Confidence scoring for each coordinate prediction

---

## (B) Design Strategy

### Architecture Overview

```
Input Image → Vision Encoder → Multi-modal Fusion → Language Decoder → Structured Output
     ↓              ↓                    ↓                  ↓                ↓
  PCB Image    Spatial Features    Fused Features    Text + Coordinates   JSON Response
```

### Component Modifications

#### 1. Vision Encoder Modifications

**Base**: Qwen-VL Vision Transformer (ViT)

**Customizations**:
- **Input Resolution**: 1024x1024 (higher than standard 448x448) for PCB detail
- **Patch Size**: 16x16 (finer granularity for small defects)
- **Multi-scale Processing**: 
  - Level 1: 1024x1024 (full resolution)
  - Level 2: 512x512 (medium features)
  - Level 3: 256x256 (contextual features)
- **Spatial Position Embeddings**: Enhanced with absolute and relative coordinates
- **Defect-Specific Features**: 
  - Edge detection branch for scratches
  - Color analysis branch for discoloration
  - Template matching branch for missing components

**Architecture**:
```python
class PCBVisionEncoder(nn.Module):
    def __init__(self):
        # Base ViT
        self.vit = QwenVL_ViT()
        
        # Multi-scale branches
        self.fpn = FeaturePyramidNetwork()
        
        # Defect-specific branches
        self.edge_branch = EdgeDetectionBranch()
        self.color_branch = ColorAnalysisBranch()
        self.template_branch = TemplateMatchingBranch()
        
        # Spatial coordinate embedding
        self.coord_embed = CoordinateEmbedding()
```

#### 2. Language Decoder Modifications

**Base**: Qwen-VL Language Model (Qwen-2B)

**Customizations**:
- **Structured Output Head**: Separate heads for:
  - Text generation (natural language response)
  - Coordinate prediction (bounding boxes)
  - Confidence scores
- **Constrained Generation**: 
  - Grammar constraints for structured responses
  - Coordinate validation (within image bounds)
  - Confidence score normalization
- **Task-Specific Prompts**: Pre-defined templates for common queries

**Architecture**:
```python
class PCBLanguageDecoder(nn.Module):
    def __init__(self):
        # Base language model
        self.lm = Qwen_LM()
        
        # Structured output heads
        self.text_head = TextGenerationHead()
        self.coord_head = CoordinateRegressionHead()
        self.confidence_head = ConfidenceHead()
        
        # Constraint validator
        self.validator = OutputValidator()
```

#### 3. Fusion Mechanism Modifications

**Base**: Cross-attention between vision and language

**Customizations**:
- **Spatial-Aware Fusion**: 
  - Cross-attention with spatial position information
  - Query-specific feature selection
  - Multi-head attention for different defect aspects
- **Query-Image Alignment**:
  - Parse query to extract spatial references ("top-left", "center", etc.)
  - Weight features based on query relevance
  - Dynamic feature selection based on query type

**Architecture**:
```python
class PCBFusion(nn.Module):
    def __init__(self):
        # Spatial-aware cross-attention
        self.spatial_cross_attn = SpatialCrossAttention()
        
        # Query parser
        self.query_parser = QueryParser()
        
        # Feature selector
        self.feature_selector = AdaptiveFeatureSelector()
```

### PCB-Specific Requirements Handling

1. **Defect Type Recognition**
   - Specialized embeddings for PCB defect vocabulary
   - Visual-text alignment for defect terminology
   - Hierarchical classification (defect category → specific type)

2. **Precision Localization**
   - Sub-pixel coordinate prediction
   - Multi-defect handling (multiple bounding boxes)
   - Defect relationship modeling (e.g., "defects near component X")

3. **Structured Response Format**
   - JSON schema enforcement
   - Coordinate system consistency (pixel coordinates)
   - Confidence score calibration

---

## (C) Optimization for <2s Inference and Offline Deployment

### Optimization Techniques

#### 1. Quantization

**Strategy**: INT8 Quantization with QAT (Quantization-Aware Training)

- **Vision Encoder**: INT8 quantization (4x size reduction, 2-3x speedup)
- **Language Decoder**: INT8 for most layers, FP16 for attention (balance speed/accuracy)
- **Expected Speedup**: 2-3x faster inference

**Implementation**:
```python
# Post-training quantization
model = quantize_model(model, dtype=torch.int8)

# Or quantization-aware training
model = prepare_qat(model)
train_with_qat(model, ...)
model = convert_to_quantized(model)
```

#### 2. Pruning

**Strategy**: Structured Pruning + Unstructured Pruning

- **Structured Pruning**: Remove entire attention heads/channels (30-40% reduction)
- **Unstructured Pruning**: Fine-grained weight pruning (10-20% additional reduction)
- **Iterative Pruning**: Gradual pruning during training

**Target Reduction**: 40-50% parameters with <2% accuracy drop

#### 3. Knowledge Distillation

**Strategy**: Teacher-Student Architecture

- **Teacher**: Full Qwen-VL-7B (high accuracy)
- **Student**: Custom compact model (2B parameters, <2s inference)
- **Distillation Loss**: 
  - Response similarity
  - Coordinate prediction accuracy
  - Confidence score calibration

**Expected**: 3-4x smaller model, 2x faster, 90%+ of teacher accuracy

#### 4. Model Architecture Optimizations

- **Flash Attention**: Memory-efficient attention (2x speedup)
- **Grouped Query Attention (GQA)**: Reduce KV cache size
- **Layer Reduction**: Remove non-critical layers (10-15% reduction)
- **Token Reduction**: Limit max tokens to 512 (from 2048)

#### 5. Inference Optimizations

- **Batch Processing**: Process multiple queries on same image
- **Caching**: Cache vision features for repeated queries
- **Early Exit**: Stop generation when confidence is high
- **TensorRT/ONNX Runtime**: Optimized inference engines

### Deployment Architecture

```
Offline Deployment Stack:
├── Model: Quantized + Pruned Qwen-VL (2B params, INT8)
├── Runtime: ONNX Runtime / TensorRT
├── Hardware: NVIDIA Jetson Orin / Intel NUC with GPU
├── Inference Time: <1.5s (with 0.5s buffer)
└── Memory: <8GB RAM, <4GB VRAM
```

### Performance Targets

| Metric | Target | Method |
|--------|--------|--------|
| Inference Time | <2s | Quantization + Pruning + Optimized Runtime |
| Model Size | <2GB | INT8 Quantization + Pruning |
| Accuracy | >90% mAP | Knowledge Distillation |
| Memory | <8GB | Model Optimization + Efficient Runtime |

---

## (D) Hallucination Mitigation

### Problem Analysis

Hallucinations in VLMs for PCB inspection manifest as:
1. **False Defect Reports**: Reporting non-existent defects
2. **Incorrect Locations**: Wrong bounding box coordinates
3. **Confidence Mismatch**: High confidence for incorrect predictions
4. **Query Misunderstanding**: Answering wrong question

### Mitigation Strategies

#### 1. Training Strategies

**A. Grounded Training**
- **Data**: Only train on image-query pairs with verified ground truth
- **Loss Function**: Penalize predictions without visual evidence
- **Verification**: Cross-check predictions with bounding box annotations

**B. Contrastive Learning**
- **Positive Pairs**: (Image, Correct Query-Answer)
- **Negative Pairs**: (Image, Incorrect Query-Answer), (Different Image, Query-Answer)
- **Objective**: Maximize similarity for positive pairs, minimize for negative

**C. Consistency Regularization**
- **Augmentation Consistency**: Same query on augmented images → same answer
- **Multi-view Consistency**: Different views of same defect → consistent description
- **Temporal Consistency**: Same defect over time → stable predictions

#### 2. Loss Functions

**A. Grounded Loss**
```python
def grounded_loss(pred, target, visual_evidence):
    # Penalize predictions without visual evidence
    evidence_mask = visual_evidence > threshold
    loss = standard_loss(pred, target)
    penalty = (1 - evidence_mask) * confidence(pred)
    return loss + lambda_penalty * penalty
```

**B. Confidence Calibration Loss**
```python
def calibration_loss(pred_confidence, actual_accuracy):
    # Ensure confidence matches actual accuracy
    return mse(pred_confidence, actual_accuracy)
```

**C. Coordinate Consistency Loss**
```python
def coordinate_consistency_loss(pred_coords, image_features):
    # Ensure coordinates align with visual features
    spatial_features = extract_features_at_coords(image_features, pred_coords)
    defect_features = extract_defect_features(image_features)
    return cosine_loss(spatial_features, defect_features)
```

#### 3. Architectural Changes

**A. Evidence Gate**
- Gate that prevents generation without sufficient visual evidence
- Threshold-based activation
- Learnable evidence scoring

**B. Confidence Head Calibration**
- Separate confidence prediction with calibration
- Temperature scaling for confidence scores
- Uncertainty quantification

**C. Multi-Modal Verification**
- Cross-check text output with visual features
- Re-verify coordinates against image
- Consistency checks between components

#### 4. Inference-Time Mitigation

**A. Confidence Thresholding**
- Only output predictions above confidence threshold (e.g., 0.7)
- Reject low-confidence predictions with "I cannot detect this defect"

**B. Coordinate Validation**
- Validate coordinates are within image bounds
- Check coordinate consistency (x1 < x2, y1 < y2)
- Verify coordinates align with visual features

**C. Response Filtering**
- Filter out responses that don't match query intent
- Validate structured output format
- Cross-reference with known defect patterns

#### 5. Training Data Strategy

**A. High-Quality Annotations**
- Human-verified annotations only
- Multiple annotators for consistency
- Regular quality audits

**B. Negative Examples**
- Include "no defect" examples
- Explicitly train on "I don't see a defect" responses
- Hard negative mining for difficult cases

**C. Diverse Query Types**
- Cover all possible query patterns
- Include ambiguous queries to teach uncertainty
- Edge cases and corner cases

### Hallucination Metrics

1. **False Positive Rate**: Defects reported but not present
2. **Coordinate Accuracy**: Percentage of correct bounding boxes
3. **Confidence Calibration Error**: Difference between confidence and accuracy
4. **Query Alignment**: Percentage of answers matching query intent

**Target Metrics**:
- False Positive Rate: <5%
- Coordinate Accuracy (IoU>0.5): >90%
- Confidence Calibration Error: <0.1
- Query Alignment: >95%

---

## (E) Training Plan

### Multi-Stage Training Approach

#### Stage 1: Vision-Language Pre-training (2-3 weeks)

**Objective**: Learn general vision-language alignment

**Data**:
- 50,000 PCB images with bounding boxes
- Generated QA pairs (see QA Generation Strategy)
- General vision-language datasets (optional, for transfer learning)

**Process**:
1. Initialize with Qwen-VL weights
2. Fine-tune vision encoder on PCB images
3. Train vision-language alignment
4. Learn coordinate token embeddings

**Metrics**: Vision-language alignment score, coordinate prediction accuracy

#### Stage 2: Task-Specific Fine-tuning (2-3 weeks)

**Objective**: Learn PCB defect detection and localization

**Data**:
- Curated QA pairs with verified answers
- Defect-specific queries
- Negative examples (no defects)

**Process**:
1. Fine-tune on defect detection tasks
2. Train coordinate regression head
3. Calibrate confidence predictions
4. Implement evidence gating

**Metrics**: mAP, coordinate accuracy, confidence calibration

#### Stage 3: Hallucination Mitigation (1-2 weeks)

**Objective**: Reduce false positives and improve reliability

**Data**:
- Hard negative examples
- Ambiguous queries
- Edge cases

**Process**:
1. Contrastive learning for negative examples
2. Confidence calibration training
3. Evidence gating training
4. Multi-view consistency training

**Metrics**: False positive rate, confidence calibration error

#### Stage 4: Optimization (1 week)

**Objective**: Optimize for inference speed

**Process**:
1. Quantization-aware training
2. Pruning and fine-tuning
3. Knowledge distillation (if needed)
4. Inference optimization

**Metrics**: Inference time, model size, accuracy retention

### QA Pair Generation Strategy

#### Challenge: 50,000 images with bounding boxes but no QA pairs

#### Solution: Multi-Stage QA Generation

**Stage 1: Template-Based Generation**

Generate QA pairs using templates:

```python
templates = [
    "What defects are present in this PCB?",
    "Where are the {defect_type} defects located?",
    "How many {defect_type} defects are there?",
    "What is the location of the defect in the {region}?",
    "Are there any defects near {component}?",
    "What type of defect is at coordinates ({x}, {y})?",
    "Describe the defect at location ({x}, {y}).",
    "What is the severity of the defect in the {region}?",
]
```

**Stage 2: LLM-Based Generation**

Use GPT-4/Claude to generate diverse queries:

```python
def generate_queries(image, bboxes):
    prompt = f"""
    Given a PCB image with defects at locations {bboxes},
    generate 10 diverse natural language questions about these defects.
    Include questions about:
    - Defect types
    - Locations
    - Counts
    - Relationships
    - Severity
    """
    queries = llm.generate(prompt)
    return queries
```

**Stage 3: Human Refinement**

- Human annotators verify and refine QA pairs
- Add edge cases and difficult queries
- Ensure natural language diversity

**Stage 4: Active Learning**

- Identify difficult cases during training
- Generate targeted QA pairs for these cases
- Iteratively improve coverage

#### QA Pair Statistics

- **Total Pairs**: ~200,000 (4 per image average)
- **Query Types**: 
  - Defect detection: 30%
  - Localization: 25%
  - Counting: 20%
  - Description: 15%
  - Relationship: 10%
- **Answer Format**: Structured JSON with coordinates and confidence

### Data Augmentation

**Image Augmentations**:
- Rotation (±5°)
- Brightness/Contrast variation
- Noise injection
- Partial occlusion
- Resolution variation

**Query Augmentations**:
- Paraphrasing (same meaning, different wording)
- Synonym replacement
- Question rephrasing
- Coordinate format variation (pixel, percentage, relative)

**Synthetic Data**:
- Generate synthetic PCB defects
- Combine real images with synthetic defects
- Domain adaptation techniques

### Evaluation Metrics

#### 1. Accuracy Metrics
- **mAP (mean Average Precision)**: Defect detection accuracy
- **Coordinate Accuracy**: IoU > 0.5 for bounding boxes
- **Query Answer Accuracy**: Correct answers to queries

#### 2. Speed Metrics
- **Inference Time**: <2s requirement
- **Throughput**: Queries per second
- **Latency Distribution**: P50, P95, P99

#### 3. Reliability Metrics
- **False Positive Rate**: <5%
- **Confidence Calibration Error**: <0.1
- **Hallucination Rate**: <3%

#### 4. Localization Metrics
- **IoU Distribution**: Mean, median IoU
- **Coordinate Error**: Mean pixel error
- **Multi-defect Accuracy**: Accuracy when multiple defects present

---

## (F) Validation

### Validation Strategy

#### 1. Counting Accuracy Validation

**Test Cases**:
- Images with 0, 1, 2, 3, 4, 5+ defects
- Multiple defects of same type
- Multiple defects of different types
- Overlapping defects

**Metrics**:
- **Exact Match Rate**: Percentage of queries with exact count
- **Count Error Distribution**: Histogram of count errors
- **Per-Type Count Accuracy**: Accuracy for each defect type

**Validation Process**:
```python
def validate_counting(model, test_set):
    results = []
    for image, query, true_count in test_set:
        response = model(image, query)
        pred_count = extract_count(response)
        results.append({
            'exact_match': pred_count == true_count,
            'error': abs(pred_count - true_count)
        })
    return compute_metrics(results)
```

#### 2. Localization Precision Validation

**Test Cases**:
- Single defect localization
- Multiple defect localization
- Small defects (<10 pixels)
- Large defects (>100 pixels)
- Edge cases (defects at image boundaries)

**Metrics**:
- **IoU Distribution**: Mean, median, P95 IoU
- **Coordinate Error**: Mean pixel error in x, y, width, height
- **Recall at IoU Thresholds**: Recall at IoU 0.3, 0.5, 0.7, 0.9
- **Precision-Recall Curve**: At different IoU thresholds

**Validation Process**:
```python
def validate_localization(model, test_set):
    results = []
    for image, query, true_bboxes in test_set:
        response = model(image, query)
        pred_bboxes = extract_bboxes(response)
        
        # Match predictions to ground truth
        ious = compute_ious(pred_bboxes, true_bboxes)
        matched = match_bboxes(ious, threshold=0.5)
        
        results.append({
            'ious': ious[matched],
            'coordinate_errors': compute_coord_errors(pred_bboxes, true_bboxes, matched)
        })
    return compute_localization_metrics(results)
```

#### 3. Hallucination Rate Validation

**Test Cases**:
- Images with no defects (should return "no defects")
- Images with defects (should not hallucinate additional defects)
- Ambiguous queries (should express uncertainty)
- Out-of-distribution images

**Metrics**:
- **False Positive Rate**: Defects reported but not present
- **False Negative Rate**: Defects present but not reported
- **Hallucination Rate**: (False Positives) / (Total Predictions)
- **Uncertainty Expression**: Percentage of queries where model expresses uncertainty

**Validation Process**:
```python
def validate_hallucination(model, test_set):
    results = {
        'false_positives': 0,
        'false_negatives': 0,
        'total_predictions': 0,
        'uncertainty_expressions': 0
    }
    
    for image, query, ground_truth in test_set:
        response = model(image, query)
        pred_defects = extract_defects(response)
        true_defects = ground_truth['defects']
        
        # Check for false positives
        for pred_defect in pred_defects:
            if not matches_any(pred_defect, true_defects):
                results['false_positives'] += 1
        
        # Check for false negatives
        for true_defect in true_defects:
            if not matches_any(true_defect, pred_defects):
                results['false_negatives'] += 1
        
        results['total_predictions'] += len(pred_defects)
        
        # Check uncertainty expression
        if 'uncertain' in response.lower() or 'cannot' in response.lower():
            results['uncertainty_expressions'] += 1
    
    results['hallucination_rate'] = results['false_positives'] / results['total_predictions']
    return results
```

### Comprehensive Validation Framework

#### Test Suite Structure

```
validation/
├── test_cases/
│   ├── counting/
│   │   ├── zero_defects/
│   │   ├── single_defect/
│   │   ├── multiple_defects/
│   │   └── edge_cases/
│   ├── localization/
│   │   ├── single_defect/
│   │   ├── multiple_defects/
│   │   ├── small_defects/
│   │   └── boundary_cases/
│   └── hallucination/
│       ├── no_defect_images/
│       ├── ambiguous_queries/
│       └── ood_images/
├── metrics/
│   ├── counting_metrics.py
│   ├── localization_metrics.py
│   └── hallucination_metrics.py
└── reports/
    └── validation_report.json
```

#### Validation Report Format

```json
{
  "counting_accuracy": {
    "exact_match_rate": 0.92,
    "mean_count_error": 0.15,
    "per_type_accuracy": {
      "scratch": 0.94,
      "misalignment": 0.91,
      "missing_component": 0.89,
      "discoloration": 0.93
    }
  },
  "localization_precision": {
    "mean_iou": 0.78,
    "median_iou": 0.82,
    "recall_at_iou_0.5": 0.91,
    "mean_coordinate_error_pixels": 3.2
  },
  "hallucination_rate": {
    "false_positive_rate": 0.03,
    "false_negative_rate": 0.05,
    "hallucination_rate": 0.02,
    "uncertainty_expression_rate": 0.15
  },
  "overall_performance": {
    "inference_time_ms": 1450,
    "model_size_mb": 1850,
    "memory_usage_gb": 6.2
  }
}
```

### Continuous Validation

1. **Automated Testing**: Run validation suite after each training epoch
2. **Regression Testing**: Ensure new changes don't degrade performance
3. **A/B Testing**: Compare model versions on held-out test set
4. **Real-world Validation**: Deploy to test environment with real queries

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)
- Set up Qwen-VL base model
- Implement custom vision encoder modifications
- Develop QA pair generation pipeline
- Create initial training infrastructure

### Phase 2: Training (Weeks 5-10)
- Stage 1: Vision-language pre-training
- Stage 2: Task-specific fine-tuning
- Stage 3: Hallucination mitigation
- Iterative improvement based on validation

### Phase 3: Optimization (Weeks 11-12)
- Quantization and pruning
- Knowledge distillation (if needed)
- Inference optimization
- Performance tuning

### Phase 4: Validation & Deployment (Weeks 13-14)
- Comprehensive validation
- Real-world testing
- Documentation
- Deployment preparation

---

## Conclusion

This design provides a comprehensive solution for custom VLM-based PCB inspection. By leveraging Qwen-VL as a foundation with significant customizations, implementing rigorous hallucination mitigation, and following a structured training approach, we can achieve:

- **<2s inference time** through optimization techniques
- **High accuracy** with >90% mAP and >90% coordinate accuracy
- **Low hallucination rate** <3% through training strategies and architectural changes
- **Offline deployment** with <2GB model size and <8GB memory

The multi-stage training approach, comprehensive QA pair generation, and rigorous validation framework ensure a robust, production-ready system for industrial quality inspection.

---

## References

1. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond (Alibaba, 2024)
2. LLaVA: Visual Instruction Tuning (Liu et al., 2023)
3. BLIP-2: Bootstrapping Language-Image Pre-training (Li et al., 2023)
4. Quantization and Pruning Techniques for Efficient Inference (Various)
5. Hallucination Mitigation in Vision-Language Models (Various)



