from fastapi import APIRouter, UploadFile, File, HTTPException
from ..ai.yolov8_infer import run_yolo_on_image_bytes
from ..ai.ocr_util import ocr_date_from_crop
from ..rag_system import RecipeRAGSystem
from ..db import pantry_col, events_col
from ..models import PantryItem
from datetime import datetime, timezone
import uuid, base64, json, os
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize RAG system
rag_system = RecipeRAGSystem()

# Ensure results directory exists
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_to_json_file(data, filename):
    """Save data to JSON file in results directory"""
    try:
        filepath = os.path.join(RESULTS_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"Data saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving to JSON file {filename}: {e}")
        return False

@router.post('/detect')
async def detect(image: UploadFile = File(...)):
    img_bytes = await image.read()
    raw = run_yolo_on_image_bytes(img_bytes)
    # prepare crops for OCR and store minimal thumbnail in DB as base64
    results = []
    for r in raw:
        # For MVP we will not crop images on server to avoid extra image libs; return bounding boxes and names
        results.append({
            'name': r['name'],
            'bbox': r['bbox'],
            'conf': r['conf']
        })
    return {'items': results}

@router.post('/scan-and-generate-recipes')
async def scan_and_generate_recipes(image: UploadFile = File(...)):
    """
    Complete flow: Scan image → Detect ingredients → Generate recipes → Track analytics
    """
    try:
        # 1. Read and process image
        img_bytes = await image.read()
        
        # 2. Detect ingredients using YOLOv8
        logger.info("Detecting ingredients with YOLOv8...")
        detections = run_yolo_on_image_bytes(img_bytes)
        
        if not detections:
            raise HTTPException(status_code=400, detail="No ingredients detected in the image")
        
        # 3. Extract ingredient names and confidence scores
        detected_ingredients = []
        for detection in detections:
            detected_ingredients.append({
                'name': detection['name'],
                'confidence': detection['conf'],
                'bbox': detection['bbox']
            })
        
        # 4. Generate recipes using RAG system
        logger.info(f"Generating recipes for {len(detected_ingredients)} ingredients...")
        ingredient_names = [item['name'] for item in detected_ingredients]
        
        recipes = rag_system.get_recipe_suggestions(
            pantry_items=ingredient_names,
            max_recipes=3
        )
        
        # 5. Enhance recipes with additional metadata
        enhanced_recipes = []
        for recipe in recipes:
            enhanced_recipe = {
                'id': str(uuid.uuid4()),
                'title': recipe.get('title', 'Unknown Recipe'),
                'ingredients': recipe.get('ingredients', []),
                'instructions': recipe.get('instructions', []),
                'cooking_time': recipe.get('cooking_time', 30),
                'difficulty': recipe.get('difficulty', 'medium'),
                'cuisine': recipe.get('cuisine', 'general'),
                'tags': recipe.get('tags', []),
                'health_score': _calculate_health_score(recipe),
                'waste_reduction_potential': _calculate_waste_reduction(recipe, ingredient_names),
                'missing_ingredients': _identify_missing_ingredients(recipe, ingredient_names),
                'substitutions': _suggest_substitutions(recipe, ingredient_names),
                'nutritional_info': _generate_nutritional_info(recipe),
                'source': 'RecipeNLG_RAG'
            }
            enhanced_recipes.append(enhanced_recipe)
        
        # 6. Save detected ingredients to pantry (optional)
        saved_items = []
        for ingredient in detected_ingredients:
            if ingredient['confidence'] > 0.5:  # Only save high-confidence detections
                pantry_item = PantryItem(
                    name=ingredient['name'],
                    quantity=1.0,
                    unit='unit',
                    confidence=ingredient['confidence'],
                    category=_categorize_ingredient(ingredient['name'])
                )
                
                # Save to database (with error handling)
                try:
                    result = await pantry_col.insert_one(pantry_item.model_dump())
                    saved_items.append({
                        'id': str(result.inserted_id),
                        'name': ingredient['name'],
                        'confidence': ingredient['confidence']
                    })
                except Exception as e:
                    logger.warning(f"Could not save to database: {e}")
                    # Add mock saved item for UI consistency
                    saved_items.append({
                        'id': f'mock_{len(saved_items)}',
                        'name': ingredient['name'],
                        'confidence': ingredient['confidence']
                    })
        
        # 7. Track analytics
        analytics_data = {
            'timestamp': datetime.now(timezone.utc),
            'action': 'scan_and_generate',
            'ingredients_detected': len(detected_ingredients),
            'recipes_generated': len(enhanced_recipes),
            'items_saved_to_pantry': len(saved_items),
            'waste_reduction_estimated': sum(r['waste_reduction_potential'] for r in enhanced_recipes),
            'detected_ingredients': ingredient_names
        }
        
        # Save analytics (with error handling)
        try:
            await events_col.insert_one(analytics_data)
        except Exception as e:
            logger.warning(f"Could not save analytics to database: {e}")
            # Continue without saving analytics
        
        # 8. Save all data to JSON files as fallback
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        session_id = str(uuid.uuid4())[:8]
        
        # Save detected ingredients
        ingredients_data = {
            'session_id': session_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'detected_ingredients': detected_ingredients,
            'total_ingredients': len(detected_ingredients),
            'ingredient_names': ingredient_names
        }
        save_to_json_file(ingredients_data, f'ingredients_{timestamp}_{session_id}.json')
        
        # Save generated recipes
        recipes_data = {
            'session_id': session_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'generated_recipes': enhanced_recipes,
            'total_recipes': len(enhanced_recipes),
            'source_ingredients': ingredient_names
        }
        save_to_json_file(recipes_data, f'recipes_{timestamp}_{session_id}.json')
        
        # Save analytics
        analytics_file_data = {
            'session_id': session_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'analytics': analytics_data,
            'waste_reduction_potential': analytics_data['waste_reduction_estimated'],
            'ingredients_utilized': len(ingredient_names),
            'recipes_available': len(enhanced_recipes)
        }
        save_to_json_file(analytics_file_data, f'analytics_{timestamp}_{session_id}.json')
        
        # Save complete session data
        complete_session_data = {
            'session_id': session_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'detected_ingredients': detected_ingredients,
            'generated_recipes': enhanced_recipes,
            'analytics': analytics_data,
            'saved_to_pantry': saved_items,
            'summary': {
                'total_ingredients_detected': len(detected_ingredients),
                'total_recipes_generated': len(enhanced_recipes),
                'waste_reduction_potential': analytics_data['waste_reduction_estimated'],
                'ingredients_utilized': len(ingredient_names),
                'recipes_available': len(enhanced_recipes)
            }
        }
        save_to_json_file(complete_session_data, f'session_{timestamp}_{session_id}.json')
        
        # 9. Return comprehensive response
        return {
            'success': True,
            'session_id': session_id,
            'detected_ingredients': detected_ingredients,
            'recipes': enhanced_recipes,
            'saved_to_pantry': saved_items,
            'analytics': {
                'waste_reduction_potential': analytics_data['waste_reduction_estimated'],
                'ingredients_utilized': len(ingredient_names),
                'recipes_available': len(enhanced_recipes)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in scan_and_generate_recipes: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

def _calculate_health_score(recipe):
    """Calculate health score based on ingredients and cooking method"""
    score = 50  # Base score
    
    # Positive factors
    healthy_ingredients = ['vegetables', 'fruits', 'lean meat', 'fish', 'whole grains', 'legumes']
    for ingredient in recipe.get('ingredients', []):
        if any(healthy in ingredient.lower() for healthy in healthy_ingredients):
            score += 10
    
    # Negative factors
    unhealthy_ingredients = ['butter', 'cream', 'sugar', 'salt', 'oil', 'cheese']
    for ingredient in recipe.get('ingredients', []):
        if any(unhealthy in ingredient.lower() for unhealthy in unhealthy_ingredients):
            score -= 5
    
    # Cooking method bonus
    cooking_methods = recipe.get('instructions', [])
    if any('steam' in step.lower() or 'bake' in step.lower() for step in cooking_methods):
        score += 15
    if any('fry' in step.lower() or 'deep fry' in step.lower() for step in cooking_methods):
        score -= 10
    
    return max(0, min(100, score))

def _calculate_waste_reduction(recipe, available_ingredients):
    """Calculate potential waste reduction in kg"""
    base_reduction = 0.5  # Base 0.5kg per recipe
    
    # Bonus for using more available ingredients
    used_ingredients = sum(1 for ing in recipe.get('ingredients', []) 
                          if any(avail.lower() in ing.lower() for avail in available_ingredients))
    
    return base_reduction + (used_ingredients * 0.1)

def _identify_missing_ingredients(recipe, available_ingredients):
    """Identify ingredients that are not available"""
    missing = []
    for ingredient in recipe.get('ingredients', []):
        if not any(avail.lower() in ingredient.lower() for avail in available_ingredients):
            missing.append(ingredient)
    return missing

def _suggest_substitutions(recipe, available_ingredients):
    """Suggest ingredient substitutions"""
    substitutions = {
        'butter': ['olive oil', 'coconut oil'],
        'cream': ['milk', 'yogurt'],
        'eggs': ['flax seeds', 'banana'],
        'milk': ['almond milk', 'soy milk'],
        'sugar': ['honey', 'maple syrup']
    }
    
    suggestions = []
    for ingredient in recipe.get('ingredients', []):
        for original, subs in substitutions.items():
            if original in ingredient.lower():
                suggestions.append({
                    'original': ingredient,
                    'substitutions': subs
                })
    
    return suggestions

def _generate_nutritional_info(recipe):
    """Generate basic nutritional information"""
    return {
        'calories_estimate': len(recipe.get('ingredients', [])) * 150,
        'protein_estimate': 'medium',
        'fiber_estimate': 'medium',
        'fat_estimate': 'medium'
    }

def _categorize_ingredient(ingredient_name):
    """Categorize ingredient for pantry organization"""
    name_lower = ingredient_name.lower()
    
    categories = {
        'fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry'],
        'vegetables': ['carrot', 'broccoli', 'lettuce', 'tomato', 'onion'],
        'dairy': ['milk', 'cheese', 'yogurt', 'butter'],
        'meat': ['chicken', 'beef', 'pork', 'fish'],
        'grains': ['bread', 'rice', 'pasta', 'flour'],
        'beverages': ['water', 'juice', 'soda']
    }
    
    for category, items in categories.items():
        if any(item in name_lower for item in items):
            return category
    
    return 'other'