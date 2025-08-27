import cv2
import numpy as np
import easyocr
import re
import dateparser
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
import openai
from ultralytics import YOLO
import os
from PIL import Image
import io
import base64
import logging
from .config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoodDetectionModel:
    def __init__(self, model_path: str = None):
        """Initialize YOLOv8 model for food detection"""
        self.model_path = model_path or Config.YOLO_MODEL_PATH
        self.confidence_threshold = Config.YOLO_CONFIDENCE_THRESHOLD
        self.image_size = Config.YOLO_IMAGE_SIZE
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 model with error handling"""
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.warning(f"YOLO model not found at {self.model_path}. Downloading...")
                self._download_model()
            
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            self.model = None
    
    def _download_model(self):
        """Download YOLOv8 model if not present"""
        try:
            logger.info("Downloading YOLOv8n model...")
            model = YOLO('yolov8n.pt')  # This will download the model
            # Save to specified path
            import shutil
            shutil.copy('yolov8n.pt', self.model_path)
            logger.info(f"Model downloaded and saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise
    
    def detect_food_items(self, image_bytes: bytes, confidence_threshold: float = None) -> List[Dict[str, Any]]:
        """Detect food items in image using YOLOv8"""
        if not self.model:
            logger.warning("YOLO model not available")
            return []
        
        confidence_threshold = confidence_threshold or self.confidence_threshold
        
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Failed to decode image")
                return []
            
            # Run detection
            results = self.model.predict(
                img, 
                conf=confidence_threshold, 
                imgsz=self.image_size,
                verbose=False
            )
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                        x1, y1, x2, y2 = [int(x) for x in box.tolist()]
                        label = result.names[int(cls)]
                        
                        detections.append({
                            'name': label,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'category': self._categorize_food(label)
                        })
            
            logger.info(f"Detected {len(detections)} food items")
            return detections
        except Exception as e:
            logger.error(f"Error in food detection: {e}")
            return []

    def _categorize_food(self, label: str) -> str:
        """Categorize detected food items"""
        label_lower = label.lower()
        
        categories = {
            'fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'tomato', 'lemon', 'lime'],
            'vegetables': ['carrot', 'broccoli', 'lettuce', 'onion', 'potato', 'cucumber', 'pepper', 'garlic'],
            'dairy': ['milk', 'cheese', 'yogurt', 'butter', 'cream', 'egg', 'eggs'],
            'meat': ['chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 'turkey', 'lamb'],
            'grains': ['bread', 'rice', 'pasta', 'cereal', 'flour', 'wheat', 'oat', 'corn'],
            'beverages': ['water', 'juice', 'soda', 'beer', 'wine', 'coffee', 'tea']
        }
        
        for category, items in categories.items():
            if any(item in label_lower for item in items):
                return category
        
        return 'other'

class ExpiryDateOCR:
    def __init__(self):
        """Initialize EasyOCR for text recognition"""
        self.languages = Config.EASYOCR_LANGUAGES
        self.gpu = Config.EASYOCR_GPU
        self.reader = None
        self._initialize_reader()
    
    def _initialize_reader(self):
        """Initialize EasyOCR reader with error handling"""
        try:
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
            logger.info(f"EasyOCR initialized successfully with languages: {self.languages}")
        except Exception as e:
            logger.error(f"Error initializing EasyOCR: {e}")
            self.reader = None
    
    def extract_expiry_date(self, image_bytes: bytes) -> Tuple[Optional[date], str]:
        """Extract expiry date from image using OCR"""
        if not self.reader:
            logger.warning("EasyOCR reader not available")
            return None, ""
        
        try:
            # Convert bytes to PIL Image
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_array = np.array(img)
            
            # Read text from image
            results = self.reader.readtext(img_array)
            text = " ".join([result[1] for result in results]).upper()
            
            # Extract date using patterns
            expiry_date = self._parse_date_patterns(text)
            
            if expiry_date:
                logger.info(f"Extracted expiry date: {expiry_date}")
            else:
                logger.info("No expiry date found in image")
            
            return expiry_date, text
        except Exception as e:
            logger.error(f"Error in OCR: {e}")
            return None, ""
    
    def _parse_date_patterns(self, text: str) -> Optional[date]:
        """Parse various date patterns from text"""
        date_patterns = [
            r"(\d{4}[-\/.]\d{1,2}[-\/.]\d{1,2})",  # YYYY-MM-DD
            r"(\d{1,2}[-\/.]\d{1,2}[-\/.]\d{2,4})",  # MM-DD-YYYY
            r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s*\d{1,2},?\s*\d{2,4}",
            r"BEST\s*BY[:\s]*([A-Z0-9\-/\s]+)",
            r"EXP[:\s]*([A-Z0-9\-/\s]+)",
            r"USE\s*BY[:\s]*([A-Z0-9\-/\s]+)",
            r"EXPIRES[:\s]*([A-Z0-9\-/\s]+)",
            r"BEST\s*BEFORE[:\s]*([A-Z0-9\-/\s]+)"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                date_text = match.group(0)
                try:
                    parsed_date = dateparser.parse(date_text)
                    if parsed_date:
                        return parsed_date.date()
                except:
                    continue
        
        return None

class ExpiryPredictor:
    def __init__(self):
        """Initialize expiry prediction model"""
        # USDA shelf-life data (simplified)
        self.shelf_life_data = {
            'milk': 7,
            'cheese': 21,
            'yogurt': 14,
            'bread': 7,
            'banana': 5,
            'apple': 14,
            'tomato': 7,
            'lettuce': 7,
            'carrot': 21,
            'potato': 30,
            'onion': 30,
            'chicken': 3,
            'beef': 5,
            'fish': 2,
            'eggs': 21,
            'orange': 14,
            'lemon': 21,
            'cucumber': 7,
            'broccoli': 7,
            'rice': 180,
            'pasta': 365,
            'flour': 365
        }
    
    def predict_expiry(self, item_name: str, category: str = None) -> Optional[date]:
        """Predict expiry date based on item name and category"""
        item_lower = item_name.lower()
        
        # Check exact matches first
        for item, days in self.shelf_life_data.items():
            if item in item_lower:
                predicted_date = datetime.now().date() + timedelta(days=days)
                logger.info(f"Predicted expiry for {item_name}: {predicted_date}")
                return predicted_date
        
        # Category-based predictions
        category_days = {
            'fruits': 7,
            'vegetables': 14,
            'dairy': 7,
            'meat': 3,
            'grains': 14,
            'beverages': 7
        }
        
        if category and category in category_days:
            predicted_date = datetime.now().date() + timedelta(days=category_days[category])
            logger.info(f"Category-based expiry prediction for {item_name}: {predicted_date}")
            return predicted_date
        
        # Default prediction
        default_date = datetime.now().date() + timedelta(days=7)
        logger.info(f"Default expiry prediction for {item_name}: {default_date}")
        return default_date

class RecipeGenerator:
    def __init__(self, openai_api_key: str = None):
        """Initialize recipe generation model"""
        self.openai_api_key = openai_api_key or Config.OPENAI_API_KEY
        self.model = Config.OPENAI_MODEL
        self.max_tokens = Config.OPENAI_MAX_TOKENS
        self.temperature = Config.OPENAI_TEMPERATURE
        
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            logger.info("OpenAI API configured for recipe generation")
        else:
            logger.warning("No OpenAI API key provided. Using fallback recipe templates.")
    
    def generate_recipes(self, pantry_items: List[str], max_recipes: int = None) -> List[Dict[str, Any]]:
        """Generate recipes based on available pantry items"""
        max_recipes = max_recipes or Config.MAX_RECIPES
        
        if not self.openai_api_key:
            logger.info("Using fallback recipe templates")
            return self._generate_fallback_recipes(pantry_items, max_recipes)
        
        try:
            logger.info(f"Generating {max_recipes} recipes using OpenAI")
            return self._generate_openai_recipes(pantry_items, max_recipes)
        except Exception as e:
            logger.error(f"Error generating recipes with OpenAI: {e}")
            logger.info("Falling back to template recipes")
            return self._generate_fallback_recipes(pantry_items, max_recipes)
    
    def _generate_openai_recipes(self, pantry_items: List[str], max_recipes: int) -> List[Dict[str, Any]]:
        """Generate recipes using OpenAI API"""
        prompt = f"""
        Generate {max_recipes} simple recipes using these ingredients: {', '.join(pantry_items)}
        
        For each recipe, provide:
        - Title
        - List of ingredients (use as many from the provided list as possible)
        - Step-by-step instructions
        - Cooking time in minutes
        - Difficulty level (easy/medium/hard)
        
        Format as JSON array with objects containing: title, ingredients[], instructions[], cooking_time, difficulty
        """
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful cooking assistant that creates simple, practical recipes."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        try:
            import json
            recipes_text = response.choices[0].message.content
            recipes = json.loads(recipes_text)
            return recipes if isinstance(recipes, list) else [recipes]
        except Exception as e:
            logger.error(f"Error parsing OpenAI response: {e}")
            return self._generate_fallback_recipes(pantry_items, max_recipes)
    
    def _generate_fallback_recipes(self, pantry_items: List[str], max_recipes: int) -> List[Dict[str, Any]]:
        """Generate simple fallback recipes"""
        recipes = []
        
        # Simple recipe templates
        templates = [
            {
                "title": "Simple Stir Fry",
                "ingredients": ["vegetables", "oil", "soy sauce"],
                "instructions": [
                    "Heat oil in a pan",
                    "Add chopped vegetables",
                    "Stir fry for 5-7 minutes",
                    "Add soy sauce and serve"
                ],
                "cooking_time": 15,
                "difficulty": "easy"
            },
            {
                "title": "Quick Salad",
                "ingredients": ["lettuce", "tomato", "cucumber", "olive oil"],
                "instructions": [
                    "Wash and chop vegetables",
                    "Mix in a bowl",
                    "Drizzle with olive oil",
                    "Season with salt and pepper"
                ],
                "cooking_time": 10,
                "difficulty": "easy"
            },
            {
                "title": "Simple Pasta",
                "ingredients": ["pasta", "tomato", "garlic", "olive oil"],
                "instructions": [
                    "Boil pasta according to package instructions",
                    "Sauté garlic in olive oil",
                    "Add chopped tomatoes",
                    "Mix with cooked pasta"
                ],
                "cooking_time": 20,
                "difficulty": "easy"
            },
            {
                "title": "Quick Omelette",
                "ingredients": ["eggs", "cheese", "vegetables"],
                "instructions": [
                    "Beat eggs in a bowl",
                    "Heat oil in a pan",
                    "Pour eggs and add cheese/vegetables",
                    "Fold and cook until done"
                ],
                "cooking_time": 10,
                "difficulty": "easy"
            },
            {
                "title": "Simple Rice Bowl",
                "ingredients": ["rice", "vegetables", "soy sauce"],
                "instructions": [
                    "Cook rice according to package instructions",
                    "Steam or sauté vegetables",
                    "Combine rice and vegetables",
                    "Add soy sauce and serve"
                ],
                "cooking_time": 25,
                "difficulty": "easy"
            }
        ]
        
        # Filter templates based on available ingredients
        for template in templates[:max_recipes]:
            available_ingredients = [ing for ing in template["ingredients"] 
                                   if any(ing in item.lower() for item in pantry_items)]
            if available_ingredients:
                recipe = template.copy()
                recipe["ingredients"] = available_ingredients
                recipes.append(recipe)
        
        logger.info(f"Generated {len(recipes)} fallback recipes")
        return recipes
