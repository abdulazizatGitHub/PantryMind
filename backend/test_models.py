#!/usr/bin/env python3
"""
Model Testing Script for Food Waste Reducer
Tests all AI models to ensure they're working correctly.
"""

import os
import sys
import time
import numpy as np
from PIL import Image
import io

def test_yolo_model():
    """Test YOLOv8 model"""
    print("🧪 Testing YOLOv8 model...")
    
    try:
        from ultralytics import YOLO
        from app.config import Config
        
        # Load model
        model = YOLO(Config.YOLO_MODEL_PATH)
        
        # Create a test image (simple colored rectangle)
        test_image = Image.new('RGB', (640, 480), color='red')
        img_array = np.array(test_image)
        
        # Run detection
        start_time = time.time()
        results = model.predict(img_array, conf=Config.YOLO_CONFIDENCE_THRESHOLD, verbose=False)
        end_time = time.time()
        
        print(f"✅ YOLOv8 test passed in {end_time - start_time:.2f}s")
        return True
        
    except Exception as e:
        print(f"❌ YOLOv8 test failed: {e}")
        return False

def test_easyocr():
    """Test EasyOCR model"""
    print("🧪 Testing EasyOCR model...")
    
    try:
        import easyocr
        from app.config import Config
        
        # Initialize reader
        reader = easyocr.Reader(Config.EASYOCR_LANGUAGES, gpu=Config.EASYOCR_GPU)
        
        # Create a test image with text
        test_image = Image.new('RGB', (300, 100), color='white')
        # Note: This is a simple test without actual text
        
        # Convert to numpy array
        img_array = np.array(test_image)
        
        # Run OCR
        start_time = time.time()
        results = reader.readtext(img_array)
        end_time = time.time()
        
        print(f"✅ EasyOCR test passed in {end_time - start_time:.2f}s")
        return True
        
    except Exception as e:
        print(f"❌ EasyOCR test failed: {e}")
        return False

def test_openai():
    """Test OpenAI API"""
    print("🧪 Testing OpenAI API...")
    
    try:
        import openai
        from app.config import Config
        
        if not Config.OPENAI_API_KEY:
            print("⚠️  No OpenAI API key found - skipping test")
            return True
        
        # Set API key
        openai.api_key = Config.OPENAI_API_KEY
        
        # Test API connection
        start_time = time.time()
        response = openai.Model.list()
        end_time = time.time()
        
        print(f"✅ OpenAI API test passed in {end_time - start_time:.2f}s")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API test failed: {e}")
        return False

def test_ai_models_integration():
    """Test AI models integration"""
    print("🧪 Testing AI models integration...")
    
    try:
        from app.ai_models import FoodDetectionModel, ExpiryDateOCR, RecipeGenerator
        
        # Initialize models
        food_detector = FoodDetectionModel()
        ocr_reader = ExpiryDateOCR()
        recipe_generator = RecipeGenerator()
        
        print("✅ AI models integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ AI models integration test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("🧪 Testing configuration...")
    
    try:
        from app.config import Config
        
        # Check essential config values
        assert Config.MONGO_URI, "MongoDB URI not configured"
        assert Config.SECRET_KEY, "Secret key not configured"
        assert Config.YOLO_MODEL_PATH, "YOLO model path not configured"
        
        print("✅ Configuration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🤖 AI Models Test Suite")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("YOLOv8 Model", test_yolo_model),
        ("EasyOCR Model", test_easyocr),
        ("OpenAI API", test_openai),
        ("AI Models Integration", test_ai_models_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"   ❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your AI models are ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
