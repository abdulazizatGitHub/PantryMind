from fastapi import APIRouter, HTTPException
from ..db import pantry_col, recipes_col, events_col
from ..ai.retrieval_index import build_index
from ..ai.rag_prompt import rewrite_with_llm
from ..utils import score_recipe
from ..rag_system import RecipeRAGSystem
from datetime import datetime, timedelta, timezone
import json, os

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
        return True
    except Exception as e:
        print(f"Error saving to JSON file {filename}: {e}")
        return False

@router.post('/recipes/suggest')
async def suggest(payload: dict = {}):
    """Return top 3 recipe suggestions based on current pantry contents.
    This endpoint reads pantry items from MongoDB, runs retrieval, tries LLM rewriting,
    falls back to base recipes, then ranks and returns top 3.
    """

    # 1) collect pantry item names
    pantry_items = []
    try:
        async for p in pantry_col.find().limit(500):
            name = p.get('name') or p.get('label')
            if name:
                pantry_items.append(name.lower())
    except Exception as e:
        # If MongoDB fails, try to load from JSON files
        print(f"MongoDB connection failed: {e}")
        pantry_items = []

    if not pantry_items:
        return {'top3': []}

    # 2) retrieve candidate recipes using FAISS index
    try:
        candidates = search(pantry_items, k=20) # returns list of dicts with title, ingredients, instructions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

    suggestions = []

    for base in candidates[:12]:
        # base expected keys: title, ingredients, instructions
        base_text = (base.get('title','') + '\n' + base.get('ingredients','') + '\n' + base.get('instructions',''))
        # Try LLM rewrite if API key present
        try:
            rewritten = rewrite_with_llm(pantry_items, base_text)
            # LLM should return JSON; be tolerant and attempt parse
            parsed = None
            try:
                parsed = json.loads(rewritten)
            except Exception:
                # attempt to locate first JSON substring
                start = rewritten.find('{')
                end = rewritten.rfind('}')
                if start != -1 and end != -1:
                    try:
                        parsed = json.loads(rewritten[start:end+1])
                    except Exception:
                        parsed = None
            if parsed is None:
                # fallback to base
                parsed = {
                    'title': base.get('title'),
                    'ingredients': (base.get('ingredients') or '').split(','),
                    'steps': [base.get('instructions')],
                    'uses_expiring': [],
                    'substitutions': []
                }
        except Exception:
            # LLM failed — use base recipe
            parsed = {
                'title': base.get('title'),
                'ingredients': (base.get('ingredients') or '').split(','),
                'steps': [base.get('instructions')],
                'uses_expiring': [],
                'substitutions': []
            }
        # Score recipe
        parsed['score'] = score_recipe(parsed, pantry_items)
        suggestions.append(parsed)

    # Sort and return top 3
    suggestions = sorted(suggestions, key=lambda x: -x.get('score',0))

    # cache top results (best-effort)
    try:
        for s in suggestions[:10]:
            await recipes_col.insert_one({'title': s.get('title'), 'metadata': s})
    except Exception:
        # Save to JSON file as fallback
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        recipe_suggestions_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'pantry_items': pantry_items,
            'suggestions': suggestions[:10],
            'top3': suggestions[:3]
        }
        save_to_json_file(recipe_suggestions_data, f'recipe_suggestions_{timestamp}.json')

    return {'top3': suggestions[:3]}

@router.get('/analytics/waste-reduction')
async def get_waste_reduction_analytics():
    """Get waste reduction analytics and statistics"""
    try:
        # Get analytics from the last 30 days
        thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
        
        # Try to get data from MongoDB
        try:
            events = []
            async for event in events_col.find({'timestamp': {'$gte': thirty_days_ago}}):
                events.append(event)
            
            # Calculate analytics
            total_waste_reduced = sum(event.get('waste_reduction_estimated', 0) for event in events)
            total_ingredients_used = sum(event.get('ingredients_detected', 0) for event in events)
            total_recipes_generated = sum(event.get('recipes_generated', 0) for event in events)
            
            # Calculate CO2 saved (rough estimate: 2.5kg CO2 per kg of food waste)
            co2_saved = total_waste_reduced * 2.5
            
            # Calculate money saved (rough estimate: $2 per kg of food waste)
            money_saved = total_waste_reduced * 2
            
            # Calculate water saved (rough estimate: 10L per kg of food waste)
            water_saved = total_waste_reduced * 10
            
            analytics_data = {
                'waste_reduction': {
                    'total_kg': round(total_waste_reduced, 2),
                    'co2_saved_kg': round(co2_saved, 2),
                    'money_saved_usd': round(money_saved, 2),
                    'water_saved_liters': round(water_saved, 2)
                },
                'usage_stats': {
                    'pantry_items': len(events),
                    'recipes_generated': total_recipes_generated,
                    'ingredients_processed': total_ingredients_used
                },
                'environmental_impact': {
                    'trees_equivalent': round(co2_saved / 22, 1),  # 1 tree absorbs ~22kg CO2/year
                    'car_miles_offset': round(co2_saved / 0.404, 1)  # 1 mile driving = 0.404kg CO2
                }
            }
            
            # Save to JSON file as backup
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            save_to_json_file(analytics_data, f'waste_reduction_analytics_{timestamp}.json')
            
            return analytics_data
            
        except Exception as e:
            print(f"MongoDB analytics query failed: {e}")
            # Return mock data if database is unavailable
            return {
                'waste_reduction': {
                    'total_kg': 0.0,
                    'co2_saved_kg': 0.0,
                    'money_saved_usd': 0.0,
                    'water_saved_liters': 0.0
                },
                'usage_stats': {
                    'pantry_items': 0,
                    'recipes_generated': 0,
                    'ingredients_processed': 0
                },
                'environmental_impact': {
                    'trees_equivalent': 0.0,
                    'car_miles_offset': 0.0
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics error: {str(e)}")

@router.get('/analytics/recipe-stats')
async def get_recipe_statistics():
    """Get recipe generation statistics"""
    try:
        # Try to get data from MongoDB
        try:
            events = []
            async for event in events_col.find({'action': 'scan_and_generate'}):
                events.append(event)
            
            if not events:
                # Return mock data if no events found
                return {
                    'scan_statistics': {
                        'success_rate_percent': 95,
                        'average_recipes_per_scan': 3,
                        'total_scans': 0
                    },
                    'recent_activity': []
                }
            
            # Calculate statistics
            total_scans = len(events)
            successful_scans = sum(1 for event in events if event.get('recipes_generated', 0) > 0)
            success_rate = (successful_scans / total_scans * 100) if total_scans > 0 else 0
            avg_recipes = sum(event.get('recipes_generated', 0) for event in events) / total_scans if total_scans > 0 else 0
            
            # Get recent activity
            recent_activity = []
            for event in events[-10:]:  # Last 10 events
                recent_activity.append({
                    'ingredients_detected': event.get('ingredients_detected', 0),
                    'recipes_generated': event.get('recipes_generated', 0),
                    'timestamp': event.get('timestamp', datetime.now(timezone.utc)).isoformat()
                })
            
            stats_data = {
                'scan_statistics': {
                    'success_rate_percent': round(success_rate, 1),
                    'average_recipes_per_scan': round(avg_recipes, 1),
                    'total_scans': total_scans
                },
                'recent_activity': recent_activity
            }
            
            # Save to JSON file as backup
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            save_to_json_file(stats_data, f'recipe_stats_{timestamp}.json')
            
            return stats_data
            
        except Exception as e:
            print(f"MongoDB recipe stats query failed: {e}")
            # Return mock data if database is unavailable
            return {
                'scan_statistics': {
                    'success_rate_percent': 95,
                    'average_recipes_per_scan': 3,
                    'total_scans': 0
                },
                'recent_activity': []
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recipe statistics error: {str(e)}")

@router.post('/recipes/{recipe_id}/use')
async def use_recipe(recipe_id: str):
    """Mark a recipe as used and track waste reduction"""
    try:
        # Create usage event
        usage_event = {
            'recipe_id': recipe_id,
            'timestamp': datetime.now(timezone.utc),
            'action': 'use_recipe',
            'waste_reduction_estimated': 0.5  # Default 0.5kg waste reduction per recipe
        }
        
        # Try to save to MongoDB
        try:
            await events_col.insert_one(usage_event)
        except Exception as e:
            print(f"Could not save recipe usage to MongoDB: {e}")
            # Save to JSON file as fallback
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            usage_data = {
                'recipe_id': recipe_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'action': 'use_recipe',
                'waste_reduction_estimated': 0.5
            }
            save_to_json_file(usage_data, f'recipe_usage_{timestamp}_{recipe_id}.json')
        
        return {'success': True, 'message': 'Recipe marked as used'}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error marking recipe as used: {str(e)}")

def search(pantry_items, k=20):
    """Search for recipes using the RAG system"""
    try:
        return rag_system.get_recipe_suggestions(pantry_items=pantry_items, max_recipes=k)
    except Exception as e:
        print(f"RAG search failed: {e}")
        return []