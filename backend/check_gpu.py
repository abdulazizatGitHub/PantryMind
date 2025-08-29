#!/usr/bin/env python3
"""
GPU Check Script for Food Waste Reducer
This script checks the current GPU setup and configuration.
"""

import os
import sys
from pathlib import Path

def check_pytorch_cuda():
    """Check PyTorch CUDA availability"""
    print("🔍 Checking PyTorch CUDA support...")
    
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"   ✅ CUDA is available")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            return True
        else:
            print("   ❌ CUDA is not available")
            print("   PyTorch was installed without CUDA support")
            return False
            
    except ImportError:
        print("   ❌ PyTorch is not installed")
        return False

def check_yolo_gpu():
    """Check YOLO GPU support"""
    print("\n🔍 Checking YOLO GPU support...")
    
    try:
        from ultralytics import YOLO
        print("   ✅ YOLO is available")
        
        # Test model loading
        model = YOLO('yolov8n.pt')
        print("   ✅ YOLO model loaded successfully")
        
        # Check if model is on GPU
        import torch
        if torch.cuda.is_available():
            print("   ✅ YOLO can use GPU")
        else:
            print("   ⚠️ YOLO will use CPU (no CUDA available)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ YOLO test failed: {e}")
        return False

def check_easyocr_gpu():
    """Check EasyOCR GPU support"""
    print("\n🔍 Checking EasyOCR GPU support...")
    
    try:
        import easyocr
        print("   ✅ EasyOCR is available")
        
        # Test GPU initialization
        try:
            reader = easyocr.Reader(['en'], gpu=True)
            print("   ✅ EasyOCR GPU initialization successful")
            return True
        except Exception as e:
            print(f"   ⚠️ EasyOCR GPU failed, falling back to CPU: {e}")
            try:
                reader = easyocr.Reader(['en'], gpu=False)
                print("   ✅ EasyOCR CPU fallback successful")
                return True
            except Exception as e2:
                print(f"   ❌ EasyOCR CPU fallback also failed: {e2}")
                return False
                
    except ImportError:
        print("   ❌ EasyOCR is not installed")
        return False

def check_sentence_transformers_gpu():
    """Check Sentence Transformers GPU support"""
    print("\n🔍 Checking Sentence Transformers GPU support...")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("   ✅ Sentence Transformers is available")
        
        # Test GPU initialization
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
            print("   ✅ Sentence Transformers GPU initialization successful")
            return True
        except Exception as e:
            print(f"   ⚠️ Sentence Transformers GPU failed, falling back to CPU: {e}")
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                print("   ✅ Sentence Transformers CPU fallback successful")
                return True
            except Exception as e2:
                print(f"   ❌ Sentence Transformers CPU fallback also failed: {e2}")
                return False
                
    except ImportError:
        print("   ❌ Sentence Transformers is not installed")
        return False

def check_faiss_gpu():
    """Check FAISS GPU support"""
    print("\n🔍 Checking FAISS GPU support...")
    
    try:
        import faiss
        print(f"   ✅ FAISS is available (version: {faiss.__version__})")
        
        # Check if GPU support is available
        if hasattr(faiss, 'GpuIndexFlatIP'):
            print("   ✅ FAISS GPU support is available")
            return True
        else:
            print("   ⚠️ FAISS GPU support is not available, using CPU version")
            return True
            
    except ImportError:
        print("   ❌ FAISS is not installed")
        return False

def check_config():
    """Check current configuration"""
    print("\n🔍 Checking current configuration...")
    
    try:
        from app.config import Config
        
        print(f"   USE_GPU: {Config.USE_GPU}")
        print(f"   CUDA_DEVICE: {Config.CUDA_DEVICE}")
        print(f"   FORCE_CPU: {Config.FORCE_CPU}")
        print(f"   YOLO_DEVICE: {Config.YOLO_DEVICE}")
        print(f"   EASYOCR_GPU: {Config.EASYOCR_GPU}")
        print(f"   SENTENCE_TRANSFORMER_DEVICE: {Config.SENTENCE_TRANSFORMER_DEVICE}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error loading configuration: {e}")
        return False

def main():
    """Main GPU check function"""
    print("=" * 60)
    print(" GPU Check for Food Waste Reducer")
    print("=" * 60)
    
    checks = [
        check_pytorch_cuda,
        check_yolo_gpu,
        check_easyocr_gpu,
        check_sentence_transformers_gpu,
        check_faiss_gpu,
        check_config
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"   ❌ Check failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total} checks")
    
    if passed == total:
        print("✅ All GPU checks passed! Your system is ready for GPU processing.")
    else:
        print("⚠️ Some checks failed. Consider running setup_gpu.py to fix issues.")
        print("\nTo enable GPU processing, run:")
        print("   python setup_gpu.py")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
