#!/usr/bin/env python3
"""
YOLOv8 Training Script for Food Detection
Fine-tunes YOLOv8 model on GroZi-120 dataset for food item detection.
"""

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml

def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_yolo_model(
    model_path: str = "yolov8n.pt",
    data_yaml: str = "data/food.yaml",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    lr: float = 1e-3,
    optimizer: str = "SGD",
    project: str = "runs/train",
    name: str = "food_yolov8",
    save_period: int = 10,
    patience: int = 20,
    device: str = "auto"
):
    """
    Train YOLOv8 model for food detection.
    
    Args:
        model_path: Path to base model (yolov8n.pt)
        data_yaml: Path to dataset configuration file
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        lr: Learning rate
        optimizer: Optimizer to use (SGD, Adam, etc.)
        project: Project directory for saving results
        name: Experiment name
        save_period: Save checkpoint every N epochs
        patience: Early stopping patience
        device: Device to use (auto, cpu, 0, 1, etc.)
    """
    
    print(f"🚀 Starting YOLOv8 training for food detection...")
    print(f"📁 Model: {model_path}")
    print(f"📊 Dataset: {data_yaml}")
    print(f"⏱️  Epochs: {epochs}")
    print(f"📦 Batch size: {batch_size}")
    print(f"🖼️  Image size: {img_size}")
    print(f"📈 Learning rate: {lr}")
    print(f"⚙️  Optimizer: {optimizer}")
    print(f"💾 Project: {project}/{name}")
    
    # Load base model
    try:
        model = YOLO(model_path)
        print(f"✅ Loaded base model: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model {model_path}: {e}")
        return False
    
    # Verify dataset configuration
    if not os.path.exists(data_yaml):
        print(f"❌ Dataset config not found: {data_yaml}")
        return False
    
    # Training configuration
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'lr0': lr,
        'optimizer': optimizer,
        'project': project,
        'name': name,
        'save_period': save_period,
        'patience': patience,
        'device': device,
        'verbose': True,
        'save': True,
        'cache': False,  # Disable caching for large datasets
        'amp': True,     # Use automatic mixed precision
        'cos_lr': True,  # Use cosine learning rate scheduler
        'close_mosaic': 10,  # Close mosaic augmentation in last 10 epochs
        'label_smoothing': 0.1,  # Label smoothing
        'mixup': 0.1,    # Mixup augmentation
        'copy_paste': 0.1,  # Copy-paste augmentation
    }
    
    # Start training
    try:
        print(f"\n🎯 Starting training...")
        results = model.train(**train_args)
        
        print(f"\n✅ Training completed successfully!")
        print(f"📊 Results saved to: {project}/{name}")
        
        # Export to ONNX format
        print(f"\n📦 Exporting model to ONNX format...")
        onnx_path = model.export(format='onnx', dynamic=True, simplify=True)
        print(f"✅ ONNX model saved to: {onnx_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for food detection")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                       help="Path to base model")
    parser.add_argument("--data", type=str, default="data/food.yaml",
                       help="Path to dataset configuration file")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--img-size", type=int, default=640,
                       help="Input image size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="SGD",
                       help="Optimizer to use")
    parser.add_argument("--project", type=str, default="runs/train",
                       help="Project directory")
    parser.add_argument("--name", type=str, default="food_yolov8",
                       help="Experiment name")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, 0, 1, etc.)")
    
    args = parser.parse_args()
    
    # Start training
    success = train_yolo_model(
        model_path=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr=args.lr,
        optimizer=args.optimizer,
        project=args.project,
        name=args.name,
        device=args.device
    )
    
    if success:
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Check results in: {args.project}/{args.name}")
    else:
        print(f"\n❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
