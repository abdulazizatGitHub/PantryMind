# 🤖 AI Models Setup Guide

This guide will help you set up all the AI models and API keys required for the AI Food Waste Reducer application.

## 📋 Prerequisites

- Python 3.8+
- Internet connection (for downloading models)
- OpenAI API key (optional, for enhanced recipe generation)
- GPU support (optional, for faster processing)

## 🚀 Quick Setup

### 1. Automated Setup (Recommended)

Run the automated setup script:

```bash
cd backend
python setup_models.py
```

This script will:
- Check Python version compatibility
- Install all required dependencies
- Download YOLOv8 model
- Set up EasyOCR
- Configure OpenAI (if API key provided)
- Create upload directories
- Test all models
- Create environment configuration file

### 2. Manual Setup

If you prefer manual setup or need to customize specific components:

## 🔧 Model Configuration

### YOLOv8 (Object Detection)

**Purpose**: Detects food items in images

**Configuration**:
```env
YOLO_MODEL_PATH=yolov8n.pt
YOLO_CONFIDENCE_THRESHOLD=0.3
YOLO_IMAGE_SIZE=640
```

**Model Options**:
- `yolov8n.pt` - Nano (fastest, ~6MB)
- `yolov8s.pt` - Small (balanced, ~22MB)
- `yolov8m.pt` - Medium (better accuracy, ~52MB)
- `yolov8l.pt` - Large (high accuracy, ~87MB)
- `yolov8x.pt` - Extra Large (best accuracy, ~136MB)

**Download manually**:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Downloads automatically
```

### EasyOCR (Text Recognition)

**Purpose**: Extracts expiry dates from food packaging

**Configuration**:
```env
EASYOCR_LANGUAGES=en
EASYOCR_GPU=False
```

**Language Options**:
- `en` - English (default)
- `en,es` - English and Spanish
- `en,fr,de` - Multiple languages

**GPU Support**:
- Set `EASYOCR_GPU=True` if you have CUDA-compatible GPU
- Requires `torch` with CUDA support

### OpenAI (Recipe Generation)

**Purpose**: Generates personalized recipes based on pantry items

**Configuration**:
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=1000
OPENAI_TEMPERATURE=0.7
```

**Model Options**:
- `gpt-3.5-turbo` - Fast, cost-effective (recommended)
- `gpt-4` - Higher quality, more expensive
- `gpt-4-turbo` - Latest GPT-4 model

**Getting API Key**:
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an account or sign in
3. Go to API Keys section
4. Create a new API key
5. Copy the key to your `.env` file

**Cost Estimation**:
- GPT-3.5-turbo: ~$0.002 per 1K tokens
- Typical recipe generation: ~500-1000 tokens
- Cost per recipe: ~$0.001-0.002

## 🔑 API Keys Setup

### 1. OpenAI API Key

**Required**: No (application works with fallback templates)
**Recommended**: Yes (for better recipe generation)

```env
OPENAI_API_KEY=sk-your-api-key-here
```

**Security Notes**:
- Never commit API keys to version control
- Use environment variables or `.env` files
- Rotate keys regularly
- Set usage limits in OpenAI dashboard

### 2. MongoDB Connection

**Required**: Yes

```env
MONGO_URI=mongodb://localhost:27017/food_waste_reducer
```

**Options**:
- Local MongoDB: `mongodb://localhost:27017/food_waste_reducer`
- MongoDB Atlas: `mongodb+srv://username:password@cluster.mongodb.net/food_waste_reducer`

## ⚙️ Environment Configuration

### Complete `.env` Example

```env
# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017/food_waste_reducer

# OpenAI API Configuration
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=1000
OPENAI_TEMPERATURE=0.7

# Flask Configuration
SECRET_KEY=your-secret-key-here-change-in-production
FLASK_DEBUG=False
TESTING=False

# YOLOv8 Model Configuration
YOLO_MODEL_PATH=yolov8n.pt
YOLO_CONFIDENCE_THRESHOLD=0.3
YOLO_IMAGE_SIZE=640

# EasyOCR Configuration
EASYOCR_LANGUAGES=en
EASYOCR_GPU=False

# Recipe Generation Settings
MAX_RECIPES=3
RECIPE_TIMEOUT=30

# Expiry Prediction Settings
EXPIRY_WARNING_DAYS=7
EXPIRY_CRITICAL_DAYS=3

# File Upload Settings
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216
```

## 🧪 Testing Models

### Test Individual Models

```python
# Test YOLOv8
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
print("✅ YOLOv8 ready")

# Test EasyOCR
import easyocr
reader = easyocr.Reader(['en'], gpu=False)
print("✅ EasyOCR ready")

# Test OpenAI
import openai
openai.api_key = "your-api-key"
response = openai.Model.list()
print("✅ OpenAI API ready")
```

### Test Complete Setup

```bash
cd backend
python -c "
from app.ai_models import FoodDetectionModel, ExpiryDateOCR, RecipeGenerator
print('✅ All models initialized successfully')
"
```

## 🚀 Performance Optimization

### GPU Acceleration

**For YOLOv8**:
```python
# Automatically uses GPU if available
model = YOLO('yolov8n.pt')
```

**For EasyOCR**:
```env
EASYOCR_GPU=True
```

**Requirements**:
- CUDA-compatible GPU
- PyTorch with CUDA support
- CUDA toolkit installed

### Model Optimization

**YOLOv8**:
- Use smaller models for faster inference
- Adjust confidence threshold
- Reduce image size for speed

**EasyOCR**:
- Limit language support to required languages
- Use GPU acceleration if available

**OpenAI**:
- Use GPT-3.5-turbo for cost efficiency
- Limit max tokens for faster responses
- Implement caching for repeated requests

## 🔒 Security Considerations

### API Key Security

1. **Environment Variables**: Always use environment variables
2. **No Hardcoding**: Never hardcode API keys in source code
3. **Access Control**: Limit API key permissions
4. **Monitoring**: Monitor API usage for unusual activity
5. **Rotation**: Regularly rotate API keys

### Model Security

1. **Model Validation**: Validate model outputs
2. **Input Sanitization**: Sanitize user inputs
3. **Rate Limiting**: Implement rate limiting for API calls
4. **Error Handling**: Handle model failures gracefully

## 🐛 Troubleshooting

### Common Issues

**YOLOv8 Download Fails**:
```bash
# Manual download
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

**EasyOCR Installation Issues**:
```bash
# Install with specific torch version
pip install torch torchvision torchaudio
pip install easyocr
```

**OpenAI API Errors**:
- Check API key validity
- Verify account has credits
- Check rate limits
- Ensure model name is correct

**Memory Issues**:
- Use smaller YOLO models
- Reduce image size
- Enable GPU acceleration
- Close other applications

### Performance Issues

**Slow Detection**:
- Use smaller YOLO model
- Reduce image resolution
- Enable GPU acceleration
- Optimize confidence threshold

**OCR Accuracy**:
- Improve image quality
- Ensure good lighting
- Use higher resolution images
- Try different language settings

## 📊 Model Performance

### Expected Performance

**YOLOv8 (yolov8n.pt)**:
- CPU: ~50-100ms per image
- GPU: ~10-20ms per image
- Accuracy: ~85-90% for common foods

**EasyOCR**:
- CPU: ~500-1000ms per image
- GPU: ~100-200ms per image
- Accuracy: ~80-90% for clear text

**OpenAI API**:
- Response time: ~2-5 seconds
- Cost: ~$0.001-0.002 per recipe
- Quality: High (with API key)

### Optimization Tips

1. **Batch Processing**: Process multiple images together
2. **Caching**: Cache model outputs for repeated items
3. **Async Processing**: Use async/await for API calls
4. **Image Preprocessing**: Optimize images before processing

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [EasyOCR Documentation](https://github.com/JaidedAI/EasyOCR)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [MongoDB Documentation](https://docs.mongodb.com/)

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review error logs in the application
3. Verify all dependencies are installed
4. Test models individually
5. Check API key validity and quotas

For additional help, please open an issue in the project repository.
