#!/usr/bin/env python3
"""
YOLOv8 Evaluation Script for Food Detection
Evaluates trained YOLOv8 model performance on test dataset.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def evaluate_model(
    model_path: str,
    data_yaml: str = "data/food.yaml",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    max_det: int = 300,
    device: str = "auto",
    save_results: bool = True,
    output_dir: str = "runs/evaluate"
):
    """
    Evaluate YOLOv8 model performance.
    
    Args:
        model_path: Path to trained model
        data_yaml: Path to dataset configuration
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        max_det: Maximum number of detections per image
        device: Device to use
        save_results: Whether to save evaluation results
        output_dir: Directory to save results
    """
    
    print(f"🔍 Evaluating YOLOv8 model: {model_path}")
    print(f"📊 Dataset: {data_yaml}")
    print(f"🎯 Confidence threshold: {conf_threshold}")
    print(f"📏 IoU threshold: {iou_threshold}")
    
    # Load model
    try:
        model = YOLO(model_path)
        print(f"✅ Loaded model: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Verify dataset configuration
    if not os.path.exists(data_yaml):
        print(f"❌ Dataset config not found: {data_yaml}")
        return None
    
    # Create output directory
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
    
    # Run validation
    try:
        print(f"\n🎯 Running validation...")
        results = model.val(
            data=data_yaml,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=max_det,
            device=device,
            save_json=save_results,
            save_hybrid=save_results,
            plots=save_results
        )
        
        print(f"✅ Validation completed!")
        
        # Extract metrics
        metrics = results.results_dict
        print(f"\n📊 Model Performance Metrics:")
        print(f"   mAP50: {metrics.get('metrics/mAP50', 'N/A'):.4f}")
        print(f"   mAP50-95: {metrics.get('metrics/mAP50-95', 'N/A'):.4f}")
        print(f"   Precision: {metrics.get('metrics/precision', 'N/A'):.4f}")
        print(f"   Recall: {metrics.get('metrics/recall', 'N/A'):.4f}")
        
        # Save detailed results
        if save_results:
            results_file = os.path.join(output_dir, "evaluation_results.json")
            with open(results_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"📄 Detailed results saved to: {results_file}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None

def analyze_predictions(
    model_path: str,
    test_images_dir: str,
    output_dir: str = "runs/evaluate",
    conf_threshold: float = 0.25,
    max_images: int = 20
):
    """
    Analyze model predictions on test images.
    
    Args:
        model_path: Path to trained model
        test_images_dir: Directory containing test images
        output_dir: Directory to save analysis results
        conf_threshold: Confidence threshold
        max_images: Maximum number of images to analyze
    """
    
    print(f"\n🔍 Analyzing predictions on test images...")
    
    # Load model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get test images
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        test_images.extend(Path(test_images_dir).glob(ext))
    
    if not test_images:
        print(f"❌ No test images found in: {test_images_dir}")
        return
    
    # Limit number of images
    test_images = test_images[:max_images]
    print(f"📸 Analyzing {len(test_images)} test images...")
    
    # Create output directory
    analysis_dir = os.path.join(output_dir, "predictions")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Analyze each image
    results = []
    for img_path in test_images:
        try:
            # Run prediction
            predictions = model.predict(
                str(img_path),
                conf=conf_threshold,
                save=True,
                project=analysis_dir,
                name=""
            )
            
            # Extract results
            for pred in predictions:
                if pred.boxes is not None:
                    boxes = pred.boxes
                    num_detections = len(boxes)
                    confidences = boxes.conf.cpu().numpy()
                    classes = boxes.cls.cpu().numpy()
                    
                    results.append({
                        'image': img_path.name,
                        'detections': num_detections,
                        'avg_confidence': float(np.mean(confidences)) if len(confidences) > 0 else 0,
                        'max_confidence': float(np.max(confidences)) if len(confidences) > 0 else 0,
                        'classes': classes.tolist()
                    })
                    
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
    
    # Save analysis results
    if results:
        analysis_file = os.path.join(output_dir, "prediction_analysis.json")
        with open(analysis_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Prediction analysis saved to: {analysis_file}")
        
        # Print summary
        total_detections = sum(r['detections'] for r in results)
        avg_confidence = np.mean([r['avg_confidence'] for r in results])
        print(f"\n📊 Prediction Summary:")
        print(f"   Total detections: {total_detections}")
        print(f"   Average confidence: {avg_confidence:.4f}")
        print(f"   Images with detections: {sum(1 for r in results if r['detections'] > 0)}/{len(results)}")

def create_evaluation_report(
    metrics: dict,
    output_dir: str = "runs/evaluate"
):
    """
    Create a comprehensive evaluation report.
    
    Args:
        metrics: Model evaluation metrics
        output_dir: Directory to save report
    """
    
    print(f"\n📋 Creating evaluation report...")
    
    # Create report content
    report = f"""
# YOLOv8 Food Detection Model Evaluation Report

## Model Performance Metrics

- **mAP50**: {metrics.get('metrics/mAP50', 'N/A'):.4f}
- **mAP50-95**: {metrics.get('metrics/mAP50-95', 'N/A'):.4f}
- **Precision**: {metrics.get('metrics/precision', 'N/A'):.4f}
- **Recall**: {metrics.get('metrics/recall', 'N/A'):.4f}

## Training Configuration

- **Base Model**: yolov8n.pt
- **Dataset**: GroZi-120 (food items)
- **Image Size**: 640x640
- **Batch Size**: 16
- **Epochs**: 50
- **Optimizer**: SGD
- **Learning Rate**: 1e-3

## Recommendations

Based on the evaluation results:

1. **Model Performance**: The model shows {'good' if metrics.get('metrics/mAP50', 0) > 0.7 else 'moderate' if metrics.get('metrics/mAP50', 0) > 0.5 else 'poor'} performance for food detection.

2. **Improvement Suggestions**:
   - {'Consider increasing training epochs' if metrics.get('metrics/mAP50', 0) < 0.6 else 'Model performance is satisfactory'}
   - {'Try data augmentation techniques' if metrics.get('metrics/recall', 0) < 0.6 else 'Good recall achieved'}
   - {'Adjust confidence threshold' if metrics.get('metrics/precision', 0) < 0.6 else 'Good precision achieved'}

3. **Deployment Readiness**: {'Model is ready for deployment' if metrics.get('metrics/mAP50', 0) > 0.6 else 'Model needs further training or optimization'}.

## Files Generated

- `evaluation_results.json`: Detailed metrics
- `prediction_analysis.json`: Test image predictions
- `predictions/`: Visualization of predictions on test images
- `confusion_matrix.png`: Confusion matrix plot
- `results.png`: Training results plots
"""
    
    # Save report
    report_file = os.path.join(output_dir, "evaluation_report.md")
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"📄 Evaluation report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model for food detection")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--data", type=str, default="data/food.yaml",
                       help="Path to dataset configuration file")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5,
                       help="IoU threshold")
    parser.add_argument("--test-images", type=str,
                       help="Directory containing test images for prediction analysis")
    parser.add_argument("--output-dir", type=str, default="runs/evaluate",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics = evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        output_dir=args.output_dir
    )
    
    if metrics is None:
        print("❌ Evaluation failed!")
        sys.exit(1)
    
    # Analyze predictions if test images provided
    if args.test_images:
        analyze_predictions(
            model_path=args.model,
            test_images_dir=args.test_images,
            output_dir=args.output_dir,
            conf_threshold=args.conf
        )
    
    # Create evaluation report
    create_evaluation_report(metrics, args.output_dir)
    
    print(f"\n🎉 Evaluation completed successfully!")
    print(f"📁 Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
