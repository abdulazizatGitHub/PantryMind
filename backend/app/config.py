import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    MONGO_URI = os.environ.get('MONGO_URI') or 'mongodb://localhost:27017/food_waste_reducer'
    MONGO_DB_NAME = os.environ.get('MONGO_DB_NAME') or 'food_waste_reducer'
    MONGO_USERNAME = os.environ.get('MONGO_USERNAME')
    MONGO_PASSWORD = os.environ.get('MONGO_PASSWORD')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # GPU Configuration
    USE_GPU = os.environ.get('USE_GPU', 'True').lower() == 'true'
    CUDA_DEVICE = os.environ.get('CUDA_DEVICE', '0')  # GPU device ID
    FORCE_CPU = os.environ.get('FORCE_CPU', 'False').lower() == 'true'  # Force CPU if needed
    
    # AI Model paths and configuration
    YOLO_MODEL_PATH = os.environ.get('YOLO_MODEL_PATH') or 'yolov8n.pt'
    YOLO_CONFIDENCE_THRESHOLD = float(os.environ.get('YOLO_CONFIDENCE_THRESHOLD', '0.3'))
    YOLO_IMAGE_SIZE = int(os.environ.get('YOLO_IMAGE_SIZE', '640'))
    YOLO_DEVICE = os.environ.get('YOLO_DEVICE', '0' if USE_GPU and not FORCE_CPU else 'cpu')
    
    # EasyOCR configuration
    EASYOCR_LANGUAGES = os.environ.get('EASYOCR_LANGUAGES', 'en').split(',')
    EASYOCR_GPU = os.environ.get('EASYOCR_GPU', 'True' if USE_GPU and not FORCE_CPU else 'False').lower() == 'true'
    
    # Sentence Transformers for RAG
    SENTENCE_TRANSFORMER_MODEL = os.environ.get('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')
    SENTENCE_TRANSFORMER_DEVICE = os.environ.get('SENTENCE_TRANSFORMER_DEVICE', 'cuda' if USE_GPU and not FORCE_CPU else 'cpu')
    FAISS_INDEX_PATH = os.environ.get('FAISS_INDEX_PATH', 'data/faiss/recipe_index.faiss')
    RECIPE_DATASET_PATH = os.environ.get('RECIPE_DATASET_PATH', 'data/faiss/recipes_metadata.pkl')
    
    # RAG Configuration
    RAG_TOP_K = int(os.environ.get('RAG_TOP_K', '5'))
    RAG_SIMILARITY_THRESHOLD = float(os.environ.get('RAG_SIMILARITY_THRESHOLD', '0.3'))
    RAG_ENABLE = os.environ.get('RAG_ENABLE', 'True').lower() == 'true'
    
    # Recipe generation settings
    MAX_RECIPES = int(os.environ.get('MAX_RECIPES', '3'))
    RECIPE_TIMEOUT = int(os.environ.get('RECIPE_TIMEOUT', '30'))  # seconds
    OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-3.5-turbo')
    OPENAI_MAX_TOKENS = int(os.environ.get('OPENAI_MAX_TOKENS', '1000'))
    OPENAI_TEMPERATURE = float(os.environ.get('OPENAI_TEMPERATURE', '0.3'))  # Lower for more consistent output
    
    # Expiry prediction settings
    EXPIRY_WARNING_DAYS = int(os.environ.get('EXPIRY_WARNING_DAYS', '7'))
    EXPIRY_CRITICAL_DAYS = int(os.environ.get('EXPIRY_CRITICAL_DAYS', '3'))
    
    # File upload settings
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
    
    # Development settings
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    TESTING = os.environ.get('TESTING', 'False').lower() == 'true'
    
    # Data directories
    DATA_DIR = os.environ.get('DATA_DIR', 'data')
    RECIPES_DIR = os.path.join(DATA_DIR, 'recipes')
    FAISS_DIR = os.path.join(DATA_DIR, 'faiss')
    MODELS_DIR = os.environ.get('MODELS_DIR', 'models')