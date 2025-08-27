#!/usr/bin/env python3
"""
AI Models Setup Script for Food Waste Reducer
This script helps set up and configure all AI models used in the application.
"""

import os
import sys
import subprocess
import shutil
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

def check_python_version():
    """Check if Python version is compatible"""
    print_step(1, "Checking Python version")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required Python packages"""
    print_step(2, "Installing Python dependencies")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def setup_yolo_model():
    """Set up YOLOv8 model"""
    print_step(3, "Setting up YOLOv8 model")
    
    model_path = os.environ.get('YOLO_MODEL_PATH', 'yolov8n.pt')
    
    if os.path.exists(model_path):
        print(f"✅ YOLO model already exists at {model_path}")
        return True
    
    try:
        print("📥 Downloading YOLOv8n model...")
        from ultralytics import YOLO
        
        # Download the model
        model = YOLO('yolov8n.pt')
        
        # Move to specified path if different
        if model_path != 'yolov8n.pt':
            shutil.move('yolov8n.pt', model_path)
        
        print(f"✅ YOLO model downloaded and saved to {model_path}")
        return True
    except Exception as e:
        print(f"❌ Error downloading YOLO model: {e}")
        return False

def setup_easyocr():
    """Set up EasyOCR"""
    print_step(4, "Setting up EasyOCR")
    
    try:
        import easyocr
        print("✅ EasyOCR is available")
        
        # Test initialization
        print("🧪 Testing EasyOCR initialization...")
        reader = easyocr.Reader(['en'], gpu=False)
        print("✅ EasyOCR initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error setting up EasyOCR: {e}")
        return False

def setup_openai():
    """Set up OpenAI configuration"""
    print_step(5, "Setting up OpenAI configuration")
    
    api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        print("⚠️  No OpenAI API key found")
        print("   The application will use fallback recipe templates")
        print("   To enable AI recipe generation, set OPENAI_API_KEY environment variable")
        return True
    
    try:
        import openai
        openai.api_key = api_key
        
        # Test API connection
        print("🧪 Testing OpenAI API connection...")
        response = openai.Model.list()
        print("✅ OpenAI API connection successful")
        return True
    except Exception as e:
        print(f"❌ Error connecting to OpenAI API: {e}")
        print("   The application will use fallback recipe templates")
        return True

def create_upload_directory():
    """Create upload directory for images"""
    print_step(6, "Creating upload directory")
    
    upload_folder = os.environ.get('UPLOAD_FOLDER', 'uploads')
    
    try:
        os.makedirs(upload_folder, exist_ok=True)
        print(f"✅ Upload directory created: {upload_folder}")
        return True
    except Exception as e:
        print(f"❌ Error creating upload directory: {e}")
        return False

def test_models():
    """Test all models"""
    print_step(7, "Testing AI models")
    
    try:
        # Test YOLO
        print("🧪 Testing YOLO model...")
        from ultralytics import YOLO
        model_path = os.environ.get('YOLO_MODEL_PATH', 'yolov8n.pt')
        model = YOLO(model_path)
        print("✅ YOLO model test passed")
        
        # Test EasyOCR
        print("🧪 Testing EasyOCR...")
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
        print("✅ EasyOCR test passed")
        
        # Test OpenAI (if available)
        if os.environ.get('OPENAI_API_KEY'):
            print("🧪 Testing OpenAI API...")
            import openai
            openai.api_key = os.environ.get('OPENAI_API_KEY')
            response = openai.Model.list()
            print("✅ OpenAI API test passed")
        
        print("✅ All model tests passed")
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    print_step(8, "Setting up environment configuration")
    
    env_file = Path('.env')
    env_example = Path('env.example')
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if env_example.exists():
        try:
            shutil.copy(env_example, env_file)
            print("✅ Created .env file from template")
            print("   Please edit .env file with your configuration")
            return True
        except Exception as e:
            print(f"❌ Error creating .env file: {e}")
            return False
    else:
        print("⚠️  No env.example file found")
        return True

def main():
    """Main setup function"""
    print_header("AI Food Waste Reducer - Model Setup")
    
    print("This script will set up all AI models and dependencies for the Food Waste Reducer application.")
    print("\nPrerequisites:")
    print("- Python 3.8+")
    print("- Internet connection (for downloading models)")
    print("- OpenAI API key (optional, for enhanced recipe generation)")
    
    # Check if running in the correct directory
    if not os.path.exists('requirements.txt'):
        print("\n❌ Please run this script from the backend directory")
        sys.exit(1)
    
    steps = [
        check_python_version,
        install_dependencies,
        setup_yolo_model,
        setup_easyocr,
        setup_openai,
        create_upload_directory,
        test_models,
        create_env_file
    ]
    
    failed_steps = []
    
    for step_func in steps:
        try:
            if not step_func():
                failed_steps.append(step_func.__name__)
        except Exception as e:
            print(f"❌ Unexpected error in {step_func.__name__}: {e}")
            failed_steps.append(step_func.__name__)
    
    print_header("Setup Complete")
    
    if failed_steps:
        print("⚠️  Some steps failed:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\nPlease check the errors above and try again.")
        return False
    else:
        print("✅ All setup steps completed successfully!")
        print("\nNext steps:")
        print("1. Edit the .env file with your configuration")
        print("2. Start MongoDB (if running locally)")
        print("3. Run the Flask application: python run.py")
        print("4. Start the frontend: npm run dev (from frontend directory)")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
