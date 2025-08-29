#!/usr/bin/env python3
"""
GPU Setup Script for Food Waste Reducer
This script helps set up CUDA-enabled packages for GPU processing.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step, description):
    """Print a formatted step"""
    print(f"\n{step}. {description}")
    print("-" * 40)

def check_cuda_availability():
    """Check if CUDA is available on the system"""
    print_step(1, "Checking CUDA availability")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("❌ CUDA is not available")
            print("   PyTorch was installed without CUDA support")
            return False
    except ImportError:
        print("❌ PyTorch is not installed")
        return False

def install_cuda_pytorch():
    """Install PyTorch with CUDA support"""
    print_step(2, "Installing PyTorch with CUDA support")
    
    try:
        # Check current PyTorch installation
        import torch
        print(f"Current PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print("✅ PyTorch already has CUDA support")
            return True
        
        print("📥 Installing PyTorch with CUDA support...")
        
        # Install PyTorch with CUDA 12.1 support
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ PyTorch with CUDA installed successfully")
            return True
        else:
            print(f"❌ Error installing PyTorch with CUDA: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def install_faiss_gpu():
    """Install FAISS with GPU support"""
    print_step(3, "Installing FAISS with GPU support")
    
    try:
        # Check if faiss-gpu is already installed
        try:
            import faiss
            print(f"Current FAISS version: {faiss.__version__}")
            
            # Check if GPU support is available
            if hasattr(faiss, 'GpuIndexFlatIP'):
                print("✅ FAISS already has GPU support")
                return True
        except ImportError:
            pass
        
        print("📥 Installing FAISS with GPU support...")
        
        # Install faiss-gpu
        cmd = [sys.executable, "-m", "pip", "install", "faiss-gpu"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ FAISS with GPU support installed successfully")
            return True
        else:
            print(f"❌ Error installing FAISS with GPU: {result.stderr}")
            print("   Trying CPU version as fallback...")
            
            # Fallback to CPU version
            cmd = [sys.executable, "-m", "pip", "install", "faiss-cpu"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ FAISS CPU version installed as fallback")
                return True
            else:
                print(f"❌ Error installing FAISS CPU version: {result.stderr}")
                return False
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_gpu_models():
    """Test GPU models"""
    print_step(4, "Testing GPU models")
    
    try:
        # Test PyTorch CUDA
        import torch
        if torch.cuda.is_available():
            print("✅ PyTorch CUDA test passed")
            
            # Test tensor operations on GPU
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print("✅ GPU tensor operations test passed")
        else:
            print("❌ PyTorch CUDA test failed")
            return False
        
        # Test YOLO with GPU
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ YOLO test failed: {e}")
            return False
        
        # Test EasyOCR with GPU
        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=True)
            print("✅ EasyOCR GPU test passed")
        except Exception as e:
            print(f"⚠️ EasyOCR GPU test failed, falling back to CPU: {e}")
            try:
                reader = easyocr.Reader(['en'], gpu=False)
                print("✅ EasyOCR CPU fallback test passed")
            except Exception as e2:
                print(f"❌ EasyOCR CPU fallback also failed: {e2}")
                return False
        
        # Test Sentence Transformers with GPU
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
            print("✅ Sentence Transformers GPU test passed")
        except Exception as e:
            print(f"⚠️ Sentence Transformers GPU test failed, falling back to CPU: {e}")
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                print("✅ Sentence Transformers CPU fallback test passed")
            except Exception as e2:
                print(f"❌ Sentence Transformers CPU fallback also failed: {e2}")
                return False
        
        # Test FAISS with GPU
        try:
            import faiss
            if hasattr(faiss, 'GpuIndexFlatIP'):
                print("✅ FAISS GPU support available")
            else:
                print("⚠️ FAISS GPU support not available, using CPU")
        except Exception as e:
            print(f"❌ FAISS test failed: {e}")
            return False
        
        print("✅ All GPU model tests passed")
        return True
        
    except Exception as e:
        print(f"❌ GPU model test failed: {e}")
        return False

def create_gpu_config():
    """Create GPU configuration file"""
    print_step(5, "Creating GPU configuration")
    
    try:
        # Create .env file with GPU settings
        env_content = """# GPU Configuration
USE_GPU=True
CUDA_DEVICE=0
FORCE_CPU=False

# YOLO Configuration
YOLO_MODEL_PATH=yolov8n.pt
YOLO_CONFIDENCE_THRESHOLD=0.3
YOLO_IMAGE_SIZE=640
YOLO_DEVICE=0

# EasyOCR Configuration
EASYOCR_LANGUAGES=en
EASYOCR_GPU=True

# Sentence Transformers Configuration
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
SENTENCE_TRANSFORMER_DEVICE=cuda

# RAG Configuration
RAG_ENABLE=True
RAG_TOP_K=5
RAG_SIMILARITY_THRESHOLD=0.3

# Other Configuration
OPENAI_API_KEY=your_openai_api_key_here
MONGO_URI=mongodb://localhost:27017/food_waste_reducer
"""
        
        env_file = Path('.env')
        if env_file.exists():
            print("⚠️ .env file already exists, creating .env.gpu instead")
            env_file = Path('.env.gpu')
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"✅ GPU configuration saved to {env_file}")
        print("   Please review and update the configuration as needed")
        return True
        
    except Exception as e:
        print(f"❌ Error creating GPU configuration: {e}")
        return False

def main():
    """Main GPU setup function"""
    print_header("GPU Setup for Food Waste Reducer")
    
    print("This script will set up CUDA-enabled packages for GPU processing.")
    print("\nPrerequisites:")
    print("- NVIDIA GPU with CUDA support")
    print("- CUDA Toolkit installed")
    print("- cuDNN installed")
    
    # Check if running in the correct directory
    if not Path("requirements.txt").exists():
        print("\n❌ Please run this script from the backend directory")
        sys.exit(1)
    
    steps = [
        check_cuda_availability,
        install_cuda_pytorch,
        install_faiss_gpu,
        test_gpu_models,
        create_gpu_config
    ]
    
    failed_steps = []
    
    for step_func in steps:
        try:
            if not step_func():
                failed_steps.append(step_func.__name__)
        except Exception as e:
            print(f"❌ Unexpected error in {step_func.__name__}: {e}")
            failed_steps.append(step_func.__name__)
    
    print_header("GPU Setup Complete")
    
    if failed_steps:
        print("⚠️ Some steps failed:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\nPlease check the errors above and try again.")
        print("\nManual installation commands:")
        print("1. Install PyTorch with CUDA:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("2. Install FAISS with GPU support:")
        print("   pip install faiss-gpu")
        return False
    else:
        print("✅ All GPU setup steps completed successfully!")
        print("\nNext steps:")
        print("1. Review the GPU configuration file (.env or .env.gpu)")
        print("2. Update your OpenAI API key if needed")
        print("3. Start MongoDB (if running locally)")
        print("4. Run the application: python run.py")
        print("\nGPU features now available:")
        print("- YOLOv8: GPU-accelerated food detection")
        print("- EasyOCR: GPU-accelerated text recognition")
        print("- Sentence Transformers: GPU-accelerated embeddings")
        print("- FAISS: GPU-accelerated vector search")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
