#!/usr/bin/env python3
"""
Force reload of RAG system to use new RecipeNLG data
"""

import sys
import os
import importlib
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

# Set the working directory to the app directory
os.chdir(app_dir)

print("🔄 Force Reloading RAG System...")

# Clear all cached modules related to RAG
modules_to_clear = [
    'rag_system', 'config', 'ai_models', 'models', 'utils'
]

for module in modules_to_clear:
    if module in sys.modules:
        print(f"🗑️ Clearing {module} from cache...")
        del sys.modules[module]

# Force reload
importlib.invalidate_caches()

try:
    print("📥 Loading fresh RAG system...")
    from app.rag_system import RecipeRAGSystem
    
    # Initialize RAG system
    rag = RecipeRAGSystem()
    
    # Get system status
    status = rag.get_system_status()
    print("\n📊 RAG Status (Fresh Load):")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Test recipe search
    print("\n🔍 Testing recipe search...")
    recipes = rag.search_recipes(['chicken', 'rice'], max_recipes=3)
    print(f"Found {len(recipes)} recipes")
    
    for i, recipe in enumerate(recipes, 1):
        print(f"\n{i}. {recipe['title']}")
        print(f"   Ingredients: {', '.join(recipe['ingredients'][:5])}...")
        print(f"   Source: {recipe.get('source', 'RecipeNLG')}")
        if 'prep_time_minutes' in recipe:
            print(f"   Prep Time: {recipe['prep_time_minutes']} min")
        if 'cook_time_minutes' in recipe:
            print(f"   Cook Time: {recipe['cook_time_minutes']} min")
        if 'servings' in recipe:
            print(f"   Servings: {recipe['servings']}")
    
    print("\n✅ RAG system force reload completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

                