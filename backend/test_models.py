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

def test_sentence_transformers():
    """Test Sentence Transformers"""
    print("🧪 Testing Sentence Transformers...")
    
    try:
        from sentence_transformers import SentenceTransformer
        from app.config import Config
        
        # Load model
        model = SentenceTransformer(Config.SENTENCE_TRANSFORMER_MODEL)
        
        # Test encoding
        test_texts = ["apple", "banana", "milk", "bread"]
        start_time = time.time()
        embeddings = model.encode(test_texts, normalize_embeddings=True)
        end_time = time.time()
        
        print(f"✅ Sentence Transformers test passed in {end_time - start_time:.2f}s")
        print(f"   Embedding shape: {embeddings.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Sentence Transformers test failed: {e}")
        return False

def test_faiss():
    """Test FAISS vector search"""
    print("🧪 Testing FAISS...")
    
    try:
        import faiss
        
        # Create test index
        dimension = 384  # all-MiniLM-L6-v2 dimension
        index = faiss.IndexFlatIP(dimension)
        
        # Create test vectors
        test_vectors = np.random.random((10, dimension)).astype('float32')
        faiss.normalize_L2(test_vectors)  # Normalize for cosine similarity
        
        # Add to index
        index.add(test_vectors)
        
        # Test search
        query = np.random.random((1, dimension)).astype('float32')
        faiss.normalize_L2(query)
        
        start_time = time.time()
        scores, indices = index.search(query, 3)
        end_time = time.time()
        
        print(f"✅ FAISS test passed in {end_time - start_time:.2f}s")
        return True
        
    except Exception as e:
        print(f"❌ FAISS test failed: {e}")
        return False

def test_rag_system():
    """Test RAG system"""
    print("🧪 Testing RAG System...")
    
    try:
        from app.rag_system import RecipeRAGSystem
        
        # Initialize RAG system
        rag_system = RecipeRAGSystem()
        
        # Test recipe search
        test_items = ["apple", "banana", "milk"]
        start_time = time.time()
        recipes = rag_system.search_recipes(test_items, max_recipes=2)
        end_time = time.time()
        
        print(f"✅ RAG System test passed in {end_time - start_time:.2f}s")
        print(f"   Found {len(recipes)} recipes")
        
        # Test system status
        status = rag_system.get_system_status()
        print(f"   RAG Status: {status}")
        return True
        
    except Exception as e:
        print(f"❌ RAG System test failed: {e}")
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
        
        # Test RAG status
        rag_status = recipe_generator.get_rag_status()
        print(f"   RAG Status: {rag_status}")
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
        assert Config.SENTENCE_TRANSFORMER_MODEL, "Sentence transformer model not configured"
        
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
        ("Sentence Transformers", test_sentence_transformers),
        ("FAISS", test_faiss),
        ("RAG System", test_rag_system),
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
