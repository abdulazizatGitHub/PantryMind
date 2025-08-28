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
from .rag_system import RecipeRAGSystem

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
        """Initialize recipe generation model with RAG system"""
        self.openai_api_key = openai_api_key or Config.OPENAI_API_KEY
        self.model = Config.OPENAI_MODEL
        self.max_tokens = Config.OPENAI_MAX_TOKENS
        self.temperature = Config.OPENAI_TEMPERATURE
        
        # Initialize RAG system
        self.rag_system = RecipeRAGSystem()
        
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            logger.info("OpenAI API configured for recipe generation")
        else:
            logger.warning("No OpenAI API key provided. Using RAG system with fallback templates.")
    
    def generate_recipes(self, pantry_items: List[str], max_recipes: int = None) -> List[Dict[str, Any]]:
        """Generate recipes using RAG system with optional LLM enhancement"""
        max_recipes = max_recipes or Config.MAX_RECIPES
        
        try:
            # Get expiring items from pantry
            expiring_items = self._get_expiring_items(pantry_items)
            
            # Use RAG system to get recipe suggestions
            logger.info(f"Generating recipes using RAG system for {len(pantry_items)} items")
            recipes = self.rag_system.get_recipe_suggestions(
                pantry_items, 
                expiring_items, 
                max_recipes
            )
            
            # Enhance with LLM if available
            if self.openai_api_key and recipes:
                logger.info("Enhancing recipes with LLM")
                recipes = self._enhance_recipes_with_llm(recipes, pantry_items, expiring_items)
            
            logger.info(f"Generated {len(recipes)} recipes")
            return recipes
            
        except Exception as e:
            logger.error(f"Error generating recipes: {e}")
            return self._generate_fallback_recipes(pantry_items, max_recipes)
    
    def _get_expiring_items(self, pantry_items: List[str]) -> List[str]:
        """Get items that are expiring soon from pantry"""
        # This would typically query the database for expiring items
        # For now, return empty list - this should be implemented in the routes
        return []
    
    def _enhance_recipes_with_llm(self, recipes: List[Dict[str, Any]], 
                                 pantry_items: List[str], 
                                 expiring_items: List[str]) -> List[Dict[str, Any]]:
        """Enhance recipes using OpenAI LLM"""
        try:
            enhanced_recipes = []
            
            for recipe in recipes:
                # Create prompt for recipe enhancement
                prompt = self._create_enhancement_prompt(recipe, pantry_items, expiring_items)
                
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful cooking assistant that enhances recipes based on available ingredients and dietary constraints."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                # Parse enhanced recipe
                enhanced_recipe = self._parse_llm_response(response.choices[0].message.content, recipe)
                enhanced_recipes.append(enhanced_recipe)
            
            return enhanced_recipes
            
        except Exception as e:
            logger.error(f"Error enhancing recipes with LLM: {e}")
            return recipes
    
    def _create_enhancement_prompt(self, recipe: Dict[str, Any], 
                                  pantry_items: List[str], 
                                  expiring_items: List[str]) -> str:
        """Create prompt for recipe enhancement"""
        expiring_text = f"Prioritize using these expiring items: {', '.join(expiring_items)}" if expiring_items else ""
        
        return f"""
        Enhance this recipe based on the available pantry items and constraints:
        
        Original Recipe: {recipe.get('title', '')}
        Ingredients: {', '.join(recipe.get('ingredients', []))}
        Instructions: {' '.join(recipe.get('instructions', []))}
        
        Available Pantry Items: {', '.join(pantry_items)}
        {expiring_text}
        
        Please enhance the recipe by:
        1. Adapting ingredients to use available pantry items
        2. Prioritizing expiring items if any
        3. Improving instructions for clarity
        4. Adding cooking tips
        5. Suggesting substitutions if needed
        
        Return the enhanced recipe in JSON format with: title, ingredients[], instructions[], cooking_time, difficulty, tips[]
        """
    
    def _parse_llm_response(self, response_text: str, original_recipe: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response and merge with original recipe"""
        try:
            import json
            enhanced_data = json.loads(response_text)
            
            # Merge with original recipe
            enhanced_recipe = original_recipe.copy()
            enhanced_recipe.update(enhanced_data)
            enhanced_recipe['source'] = 'rag_enhanced_llm'
            
            return enhanced_recipe
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return original_recipe
    
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
                recipe["source"] = "fallback_template"
                recipes.append(recipe)
        
        logger.info(f"Generated {len(recipes)} fallback recipes")
        return recipes
    
    def get_rag_status(self) -> Dict[str, Any]:
        """Get RAG system status"""
        return self.rag_system.get_system_status()
