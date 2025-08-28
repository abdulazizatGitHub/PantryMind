#!/usr/bin/env python3
"""
YOLOv8 Training Pipeline for Food Detection
Complete pipeline for fine-tuning YOLOv8 on GroZi-120 dataset.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
import json
import yaml

class YOLOTrainingPipeline:
    def __init__(self, config_path: str = "training_config.yaml"):
        """
        Initialize the training pipeline.
        
        Args:
            config_path: Path to training configuration file
        """
        self.config_path = config_path
        self.config = self.load_config()
        self.setup_directories()
    
    def load_config(self) -> dict:
        """Load training configuration."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'dataset': {
                    'source_path': 'data/GroZi-120',
                    'output_path': 'data/GroZi-120-yolo',
                    'train_split': 0.8,
                    'val_split': 0.1
                },
                'training': {
                    'base_model': 'yolov8n.pt',
                    'epochs': 50,
                    'batch_size': 16,
                    'img_size': 640,
                    'lr': 1e-3,
                    'optimizer': 'SGD',
                    'project': 'runs/train',
                    'name': 'food_yolov8',
                    'device': 'auto'
                },
                'evaluation': {
                    'conf_threshold': 0.25,
                    'iou_threshold': 0.5,
                    'output_dir': 'runs/evaluate'
                }
            }
    
    def setup_directories(self):
        """Create necessary directories."""
        dirs = [
            'data/GroZi-120-yolo',
            'runs/train',
            'runs/evaluate',
            'models'
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"📁 Created directory: {dir_path}")
    
    def prepare_dataset(self) -> bool:
        """Prepare the GroZi-120 dataset for training."""
        print("\n🔧 Step 1: Preparing Dataset")
        print("=" * 50)
        
        try:
            # Check if dataset is already prepared
            if os.path.exists(self.config['dataset']['output_path']):
                print(f"✅ Dataset already prepared at: {self.config['dataset']['output_path']}")
                return True
            
            # Run dataset preparation script
            cmd = [
                sys.executable, 'scripts/prepare_grozi_dataset.py',
                '--dataset-path', self.config['dataset']['source_path'],
                '--output-path', self.config['dataset']['output_path'],
                '--train-split', str(self.config['dataset']['train_split']),
                '--val-split', str(self.config['dataset']['val_split'])
            ]
            
            print(f"🔄 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Dataset preparation completed successfully!")
                return True
            else:
                print(f"❌ Dataset preparation failed:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error during dataset preparation: {e}")
            return False
    
    def update_dataset_config(self) -> bool:
        """Update the dataset configuration file."""
        print("\n📝 Step 2: Updating Dataset Configuration")
        print("=" * 50)
        
        try:
            # Update food.yaml with correct paths
            yaml_content = f"""# YOLOv8 Dataset Configuration for GroZi-120 Food Detection
path: {self.config['dataset']['output_path']}  # Dataset root directory
train: images/train  # Train images (relative to 'path')
val: images/val      # Validation images (relative to 'path')
test: images/test    # Test images (relative to 'path')

# Classes
names:
  0: food_item  # Generic food item class

# Number of classes
nc: 1  # Update this based on your class count
"""
            
            with open('data/food.yaml', 'w') as f:
                f.write(yaml_content)
            
            print("✅ Dataset configuration updated!")
            return True
            
        except Exception as e:
            print(f"❌ Error updating dataset config: {e}")
            return False
    
    def train_model(self) -> bool:
        """Train the YOLOv8 model."""
        print("\n🚀 Step 3: Training YOLOv8 Model")
        print("=" * 50)
        
        try:
            # Run training script
            cmd = [
                sys.executable, 'scripts/train_yolo.py',
                '--model', self.config['training']['base_model'],
                '--data', 'data/food.yaml',
                '--epochs', str(self.config['training']['epochs']),
                '--batch-size', str(self.config['training']['batch_size']),
                '--img-size', str(self.config['training']['img_size']),
                '--lr', str(self.config['training']['lr']),
                '--optimizer', self.config['training']['optimizer'],
                '--project', self.config['training']['project'],
                '--name', self.config['training']['name'],
                '--device', self.config['training']['device']
            ]
            
            print(f"🔄 Running: {' '.join(cmd)}")
            print(f"⏱️  Training will take approximately {self.config['training']['epochs'] * 2} minutes...")
            
            # Run training with real-time output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            for line in process.stdout:
                print(line.rstrip())
            
            process.wait()
            
            if process.returncode == 0:
                print("✅ Training completed successfully!")
                return True
            else:
                print("❌ Training failed!")
                return False
                
        except Exception as e:
            print(f"❌ Error during training: {e}")
            return False
    
    def evaluate_model(self) -> bool:
        """Evaluate the trained model."""
        print("\n🔍 Step 4: Evaluating Model")
        print("=" * 50)
        
        try:
            # Find the best model
            model_path = f"runs/train/{self.config['training']['name']}/weights/best.pt"
            if not os.path.exists(model_path):
                print(f"❌ Best model not found at: {model_path}")
                return False
            
            # Run evaluation script
            cmd = [
                sys.executable, 'scripts/evaluate_yolo.py',
                '--model', model_path,
                '--data', 'data/food.yaml',
                '--conf', str(self.config['evaluation']['conf_threshold']),
                '--iou', str(self.config['evaluation']['iou_threshold']),
                '--output-dir', self.config['evaluation']['output_dir'],
                '--device', self.config['training']['device']
            ]
            
            print(f"🔄 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Evaluation completed successfully!")
                print(result.stdout)
                return True
            else:
                print(f"❌ Evaluation failed:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            return False
    
    def export_models(self) -> bool:
        """Export models to different formats."""
        print("\n📦 Step 5: Exporting Models")
        print("=" * 50)
        
        try:
            from ultralytics import YOLO
            
            # Load best model
            model_path = f"runs/train/{self.config['training']['name']}/weights/best.pt"
            if not os.path.exists(model_path):
                print(f"❌ Best model not found at: {model_path}")
                return False
            
            model = YOLO(model_path)
            
            # Export to different formats
            export_formats = ['onnx', 'torchscript', 'tflite']
            
            for fmt in export_formats:
                try:
                    print(f"🔄 Exporting to {fmt.upper()} format...")
                    export_path = model.export(format=fmt, dynamic=True, simplify=True)
                    print(f"✅ {fmt.upper()} model saved to: {export_path}")
                except Exception as e:
                    print(f"⚠️  Failed to export to {fmt}: {e}")
            
            # Copy best model to models directory
            import shutil
            final_model_path = f"models/food_detection_best.pt"
            shutil.copy2(model_path, final_model_path)
            print(f"✅ Best model copied to: {final_model_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during model export: {e}")
            return False
    
    def create_training_report(self) -> bool:
        """Create a comprehensive training report."""
        print("\n📋 Step 6: Creating Training Report")
        print("=" * 50)
        
        try:
            # Load evaluation results
            eval_file = f"{self.config['evaluation']['output_dir']}/evaluation_results.json"
            metrics = {}
            if os.path.exists(eval_file):
                with open(eval_file, 'r') as f:
                    metrics = json.load(f)
            
            # Create report
            report = f"""# YOLOv8 Food Detection Training Report

## Training Configuration

- **Base Model**: {self.config['training']['base_model']}
- **Dataset**: GroZi-120 (food items)
- **Training Epochs**: {self.config['training']['epochs']}
- **Batch Size**: {self.config['training']['batch_size']}
- **Image Size**: {self.config['training']['img_size']}x{self.config['training']['img_size']}
- **Learning Rate**: {self.config['training']['lr']}
- **Optimizer**: {self.config['training']['optimizer']}

## Model Performance

- **mAP50**: {metrics.get('metrics/mAP50', 'N/A'):.4f if isinstance(metrics.get('metrics/mAP50'), (int, float)) else 'N/A'}
- **mAP50-95**: {metrics.get('metrics/mAP50-95', 'N/A'):.4f if isinstance(metrics.get('metrics/mAP50-95'), (int, float)) else 'N/A'}
- **Precision**: {metrics.get('metrics/precision', 'N/A'):.4f if isinstance(metrics.get('metrics/precision'), (int, float)) else 'N/A'}
- **Recall**: {metrics.get('metrics/recall', 'N/A'):.4f if isinstance(metrics.get('metrics/recall'), (int, float)) else 'N/A'}

## Files Generated

### Models
- `runs/train/{self.config['training']['name']}/weights/best.pt` - Best model weights
- `runs/train/{self.config['training']['name']}/weights/last.pt` - Last model weights
- `models/food_detection_best.pt` - Best model (copied)

### Exports
- `runs/train/{self.config['training']['name']}/weights/best.onnx` - ONNX format
- `runs/train/{self.config['training']['name']}/weights/best.torchscript` - TorchScript format
- `runs/train/{self.config['training']['name']}/weights/best.tflite` - TensorFlow Lite format

### Results
- `runs/train/{self.config['training']['name']}/results.png` - Training curves
- `runs/train/{self.config['training']['name']}/confusion_matrix.png` - Confusion matrix
- `{self.config['evaluation']['output_dir']}/evaluation_results.json` - Detailed metrics
- `{self.config['evaluation']['output_dir']}/evaluation_report.md` - Evaluation report

## Usage

### Inference with Python
```python
from ultralytics import YOLO

# Load model
model = YOLO('models/food_detection_best.pt')

# Run inference
results = model.predict('path/to/image.jpg', conf=0.25)
```

### Command Line Inference
```bash
python scripts/inference_yolo.py --model models/food_detection_best.pt --source path/to/image.jpg
```

### Webcam Detection
```bash
python scripts/inference_yolo.py --model models/food_detection_best.pt --source webcam
```

## Training Completed

Training completed successfully on {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            # Save report
            report_file = f"runs/train/{self.config['training']['name']}/training_report.md"
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(f"✅ Training report saved to: {report_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating training report: {e}")
            return False
    
    def run_pipeline(self) -> bool:
        """Run the complete training pipeline."""
        print("🎯 YOLOv8 Food Detection Training Pipeline")
        print("=" * 60)
        print(f"📅 Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        steps = [
            ("Dataset Preparation", self.prepare_dataset),
            ("Update Dataset Config", self.update_dataset_config),
            ("Model Training", self.train_model),
            ("Model Evaluation", self.evaluate_model),
            ("Model Export", self.export_models),
            ("Create Report", self.create_training_report)
        ]
        
        for step_name, step_func in steps:
            if not step_func():
                print(f"\n❌ Pipeline failed at step: {step_name}")
                return False
        
        print("\n🎉 Training Pipeline Completed Successfully!")
        print(f"📅 Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Check results in: runs/train/{self.config['training']['name']}")
        print(f"📄 Training report: runs/train/{self.config['training']['name']}/training_report.md")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Food Detection Training Pipeline")
    parser.add_argument("--config", type=str, default="training_config.yaml",
                       help="Path to training configuration file")
    parser.add_argument("--skip-dataset", action="store_true",
                       help="Skip dataset preparation step")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training step (for evaluation only)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = YOLOTrainingPipeline(args.config)
    
    # Run pipeline
    success = pipeline.run_pipeline()
    
    if not success:
        print("\n❌ Pipeline failed!")
        sys.exit(1)
    
    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
