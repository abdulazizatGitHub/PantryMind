# YOLOv8 Food Detection Training Guide

This guide covers fine-tuning YOLOv8 for food item detection using the GroZi-120 dataset.

## 🎯 Overview

The training pipeline fine-tunes YOLOv8 on the GroZi-120 dataset to detect food items in images. The pipeline includes:

- **Dataset Preparation**: Convert GroZi-120 to YOLO format
- **Model Training**: Fine-tune YOLOv8n with specified parameters
- **Model Evaluation**: Assess performance on test set
- **Model Export**: Export to ONNX, TorchScript, and TensorFlow Lite formats
- **Comprehensive Reporting**: Generate training and evaluation reports

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended for faster training)
- 8GB+ RAM
- 10GB+ free disk space

### Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```

## 🗂️ Dataset Structure

The GroZi-120 dataset should be organized as follows:
```
data/GroZi-120/
├── 1/
│   ├── video/
│   │   ├── video1.png
│   │   ├── video2.png
│   │   └── ...
│   ├── coordinates.txt
│   └── info.txt
├── 2/
│   └── ...
└── 120/
    └── ...
```

## 🚀 Quick Start

### 1. Run Complete Training Pipeline

```bash
# Run the complete training pipeline
python scripts/train_pipeline.py
```

This will:
- Prepare the GroZi-120 dataset
- Train YOLOv8n for 50 epochs
- Evaluate the model
- Export to multiple formats
- Generate comprehensive reports

### 2. Individual Steps

#### Dataset Preparation
```bash
python scripts/prepare_grozi_dataset.py \
    --dataset-path data/GroZi-120 \
    --output-path data/GroZi-120-yolo \
    --train-split 0.8 \
    --val-split 0.1
```

#### Model Training
```bash
python scripts/train_yolo.py \
    --model yolov8n.pt \
    --data data/food.yaml \
    --epochs 50 \
    --batch-size 16 \
    --img-size 640 \
    --lr 1e-3 \
    --optimizer SGD \
    --project runs/train \
    --name food_yolov8
```

#### Model Evaluation
```bash
python scripts/evaluate_yolo.py \
    --model runs/train/food_yolov8/weights/best.pt \
    --data data/food.yaml \
    --conf 0.25 \
    --iou 0.5
```

#### Inference
```bash
# Image inference
python scripts/inference_yolo.py \
    --model models/food_detection_best.pt \
    --source path/to/image.jpg \
    --output results.jpg

# Webcam inference
python scripts/inference_yolo.py \
    --model models/food_detection_best.pt \
    --source webcam

# Video inference
python scripts/inference_yolo.py \
    --model models/food_detection_best.pt \
    --source path/to/video.mp4 \
    --output results.mp4
```

## ⚙️ Configuration

### Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Base Model | `yolov8n.pt` | Pre-trained YOLOv8 nano model |
| Epochs | 50 | Number of training epochs |
| Batch Size | 16 | Training batch size |
| Image Size | 640 | Input image resolution |
| Learning Rate | 1e-3 | Initial learning rate |
| Optimizer | SGD | Stochastic Gradient Descent |
| Device | auto | Auto-detect GPU/CPU |

### Dataset Configuration

The dataset configuration is defined in `data/food.yaml`:

```yaml
path: data/GroZi-120-yolo
train: images/train
val: images/val
test: images/test

names:
  0: food_item

nc: 1
```

## 📊 Expected Results

### Training Metrics
- **mAP50**: 0.7+ (good performance)
- **mAP50-95**: 0.4+ (moderate performance)
- **Precision**: 0.6+ (good precision)
- **Recall**: 0.6+ (good recall)

### Training Time
- **GPU (RTX 3080)**: ~30-45 minutes
- **CPU**: ~3-5 hours
- **Cloud GPU**: ~15-30 minutes

## 📁 Output Structure

After training, you'll find the following structure:

```
runs/
├── train/
│   └── food_yolov8/
│       ├── weights/
│       │   ├── best.pt          # Best model weights
│       │   ├── last.pt          # Last model weights
│       │   ├── best.onnx        # ONNX format
│       │   ├── best.torchscript # TorchScript format
│       │   └── best.tflite      # TensorFlow Lite format
│       ├── results.png          # Training curves
│       ├── confusion_matrix.png # Confusion matrix
│       └── training_report.md   # Training report
└── evaluate/
    ├── evaluation_results.json  # Detailed metrics
    ├── evaluation_report.md     # Evaluation report
    └── predictions/             # Test predictions

models/
└── food_detection_best.pt       # Best model (copied)
```

## 🔧 Customization

### Modify Training Parameters

Create a custom configuration file `training_config.yaml`:

```yaml
dataset:
  source_path: data/GroZi-120
  output_path: data/GroZi-120-yolo
  train_split: 0.8
  val_split: 0.1

training:
  base_model: yolov8n.pt
  epochs: 100
  batch_size: 32
  img_size: 640
  lr: 5e-4
  optimizer: Adam
  project: runs/train
  name: food_yolov8_custom
  device: 0

evaluation:
  conf_threshold: 0.25
  iou_threshold: 0.5
  output_dir: runs/evaluate
```

Run with custom config:
```bash
python scripts/train_pipeline.py --config training_config.yaml
```

### Add More Classes

To detect specific food types instead of generic "food_item":

1. Update `data/food.yaml`:
```yaml
names:
  0: apple
  1: banana
  2: milk
  3: bread
  # Add more classes as needed

nc: 4  # Update class count
```

2. Modify the dataset preparation script to assign specific class IDs based on item directories.

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
python scripts/train_yolo.py --batch-size 8

# Use CPU training
python scripts/train_yolo.py --device cpu
```

#### 2. Dataset Not Found
```bash
# Check dataset path
ls data/GroZi-120

# Verify dataset structure
python scripts/prepare_grozi_dataset.py --dataset-path data/GroZi-120
```

#### 3. Training Stuck
```bash
# Check GPU usage
nvidia-smi

# Monitor training logs
tail -f runs/train/food_yolov8/train.log
```

#### 4. Poor Performance
- Increase training epochs: `--epochs 100`
- Adjust learning rate: `--lr 5e-4`
- Use data augmentation
- Check dataset quality

### Performance Optimization

#### GPU Training
```bash
# Use specific GPU
python scripts/train_yolo.py --device 0

# Enable mixed precision
# (Already enabled by default in ultralytics)
```

#### CPU Training
```bash
# Use CPU with optimized settings
python scripts/train_yolo.py --device cpu --batch-size 8
```

## 📈 Monitoring Training

### Real-time Monitoring
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training progress
tail -f runs/train/food_yolov8/train.log
```

### TensorBoard (Optional)
```bash
# Install tensorboard
pip install tensorboard

# Start tensorboard
tensorboard --logdir runs/train
```

## 🔄 Integration with Main Application

After training, update the main application to use the fine-tuned model:

```python
# In app/ai_models.py
class FoodDetectionModel:
    def __init__(self):
        # Use fine-tuned model instead of base model
        self.model = YOLO('models/food_detection_best.pt')
```

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [GroZi-120 Dataset Paper](https://ieeexplore.ieee.org/document/1641958)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)

## 🤝 Contributing

To improve the training pipeline:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This training pipeline is part of the AI Food Waste Reducer project.
