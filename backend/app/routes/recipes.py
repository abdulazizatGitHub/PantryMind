from flask import Blueprint, jsonify, request
from datetime import datetime
from .. import mongo
from ..ai_models import RecipeGenerator
from ..models import Recipe, UserInteraction
from ..config import Config

bp = Blueprint('recipes', __name__)

# Initialize recipe generator
recipe_generator = RecipeGenerator(Config.OPENAI_API_KEY)

@bp.route('/suggest', methods=['POST'])
def suggest_recipes():
    """Generate recipe suggestions based on pantry items"""
    try:
        data = request.get_json()
        pantry_items = data.get('pantry_items', [])
        max_recipes = data.get('max_recipes', Config.MAX_RECIPES)
        
        if not pantry_items:
            return jsonify({'error': 'No pantry items provided'}), 400
        
        # Get item names from pantry
        item_names = []
        for item_id in pantry_items:
            item = mongo.db.pantry_items.find_one({'_id': item_id})
            if item:
                item_names.append(item['name'])
        
        # Generate recipes
        recipes = recipe_generator.generate_recipes(item_names, max_recipes)
        
        # Save recipes to database
        saved_recipes = []
        for recipe_data in recipes:
            recipe = Recipe(
                title=recipe_data['title'],
                ingredients=recipe_data['ingredients'],
                instructions=recipe_data['instructions'],
                cooking_time=recipe_data.get('cooking_time', 30),
                difficulty=recipe_data.get('difficulty', 'medium')
            )
            
            result = mongo.db.recipes.insert_one(recipe.to_dict())
            recipe_data['id'] = str(result.inserted_id)
            saved_recipes.append(recipe_data)
        
        # Log interaction
        interaction = UserInteraction(
            action='generate_recipes',
            metadata={
                'pantry_items_count': len(item_names),
                'recipes_generated': len(saved_recipes)
            }
        )
        mongo.db.user_interactions.insert_one(interaction.to_dict())
        
        return jsonify({
            'recipes': saved_recipes,
            'pantry_items_used': item_names
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/suggest-from-pantry', methods=['GET'])
def suggest_from_pantry():
    """Generate recipe suggestions from all pantry items"""
    try:
        # Get all pantry items
        pantry_items = list(mongo.db.pantry_items.find())
        
        if not pantry_items:
            return jsonify({'recipes': [], 'message': 'No items in pantry'})
        
        # Extract item names
        item_names = [item['name'] for item in pantry_items]
        
        # Generate recipes
        recipes = recipe_generator.generate_recipes(item_names, Config.MAX_RECIPES)
        
        # Save recipes to database
        saved_recipes = []
        for recipe_data in recipes:
            recipe = Recipe(
                title=recipe_data['title'],
                ingredients=recipe_data['ingredients'],
                instructions=recipe_data['instructions'],
                cooking_time=recipe_data.get('cooking_time', 30),
                difficulty=recipe_data.get('difficulty', 'medium')
            )
            
            result = mongo.db.recipes.insert_one(recipe.to_dict())
            recipe_data['id'] = str(result.inserted_id)
            saved_recipes.append(recipe_data)
        
        # Log interaction
        interaction = UserInteraction(
            action='generate_recipes_from_pantry',
            metadata={
                'pantry_items_count': len(item_names),
                'recipes_generated': len(saved_recipes)
            }
        )
        mongo.db.user_interactions.insert_one(interaction.to_dict())
        
        return jsonify({
            'recipes': saved_recipes,
            'pantry_items_used': item_names
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/recipes', methods=['GET'])
def get_recipes():
    """Get all saved recipes"""
    try:
        recipes = list(mongo.db.recipes.find().sort('created_at', -1))
        
        # Convert ObjectId to string
        for recipe in recipes:
            recipe['_id'] = str(recipe['_id'])
            recipe['created_at'] = recipe['created_at'].isoformat()
        
        return jsonify({'recipes': recipes})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/recipes/<recipe_id>', methods=['GET'])
def get_recipe(recipe_id):
    """Get a specific recipe"""
    try:
        from bson import ObjectId
        recipe = mongo.db.recipes.find_one({'_id': ObjectId(recipe_id)})
        
        if not recipe:
            return jsonify({'error': 'Recipe not found'}), 404
        
        recipe['_id'] = str(recipe['_id'])
        recipe['created_at'] = recipe['created_at'].isoformat()
        
        return jsonify({'recipe': recipe})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/recipes/<recipe_id>/use', methods=['POST'])
def use_recipe(recipe_id):
    """Mark recipe as used and log waste reduction"""
    try:
        from bson import ObjectId
        
        # Log interaction
        interaction = UserInteraction(
            action='use_recipe',
            recipe_id=recipe_id,
            metadata={'timestamp': datetime.utcnow().isoformat()}
        )
        mongo.db.user_interactions.insert_one(interaction.to_dict())
        
        # Log waste reduction
        waste_interaction = UserInteraction(
            action='waste_reduced',
            recipe_id=recipe_id,
            metadata={'amount_kg': 0.5}  # Estimate 0.5kg waste reduced
        )
        mongo.db.user_interactions.insert_one(waste_interaction.to_dict())
        
        return jsonify({'message': 'Recipe used successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
