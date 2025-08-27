import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    MONGO_URI = os.environ.get('MONGO_URI') or 'mongodb://localhost:27017/food_waste_reducer'
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # AI Model paths and configuration
    YOLO_MODEL_PATH = os.environ.get('YOLO_MODEL_PATH') or 'yolov8n.pt'
    YOLO_CONFIDENCE_THRESHOLD = float(os.environ.get('YOLO_CONFIDENCE_THRESHOLD', '0.3'))
    YOLO_IMAGE_SIZE = int(os.environ.get('YOLO_IMAGE_SIZE', '640'))
    
    # EasyOCR configuration
    EASYOCR_LANGUAGES = os.environ.get('EASYOCR_LANGUAGES', 'en').split(',')
    EASYOCR_GPU = os.environ.get('EASYOCR_GPU', 'False').lower() == 'true'
    
    # Recipe generation settings
    MAX_RECIPES = int(os.environ.get('MAX_RECIPES', '3'))
    RECIPE_TIMEOUT = int(os.environ.get('RECIPE_TIMEOUT', '30'))  # seconds
    OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-3.5-turbo')
    OPENAI_MAX_TOKENS = int(os.environ.get('OPENAI_MAX_TOKENS', '1000'))
    OPENAI_TEMPERATURE = float(os.environ.get('OPENAI_TEMPERATURE', '0.7'))
    
    # Expiry prediction settings
    EXPIRY_WARNING_DAYS = int(os.environ.get('EXPIRY_WARNING_DAYS', '7'))
    EXPIRY_CRITICAL_DAYS = int(os.environ.get('EXPIRY_CRITICAL_DAYS', '3'))
    
    # File upload settings
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
    
    # Development settings
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    TESTING = os.environ.get('TESTING', 'False').lower() == 'true'