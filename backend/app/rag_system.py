"""
RAG (Retrieval-Augmented Generation) System for Recipe Suggestions
Uses sentence-transformers and FAISS for efficient recipe retrieval
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import logging
from datetime import datetime, timedelta
from .config import Config

logger = logging.getLogger(__name__)

class RecipeRAGSystem:
    def __init__(self):
        """Initialize the RAG system for recipe retrieval"""
        self.sentence_transformer = None
        self.faiss_index = None
        self.recipes_data = None
        self.recipe_embeddings = None
        self._initialize_rag_system()
    
    def _initialize_rag_system(self):
        """Initialize the RAG system components"""
        try:
            # Initialize sentence transformer
            logger.info(f"Loading sentence transformer model: {Config.SENTENCE_TRANSFORMER_MODEL}")
            self.sentence_transformer = SentenceTransformer(Config.SENTENCE_TRANSFORMER_MODEL)
            
            # Load or create FAISS index
            self._load_or_create_faiss_index()
            
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            self.sentence_transformer = None
            self.faiss_index = None
    
    def _load_or_create_faiss_index(self):
        """Load existing FAISS index or create new one"""
        try:
            # Create directories if they don't exist
            os.makedirs(Config.FAISS_DIR, exist_ok=True)
            os.makedirs(Config.RECIPES_DIR, exist_ok=True)
            
            # Check if FAISS index exists
            if os.path.exists(Config.FAISS_INDEX_PATH):
                logger.info("Loading existing FAISS index")
                self.faiss_index = faiss.read_index(Config.FAISS_INDEX_PATH)
                
                # Load recipe data
                recipes_pickle_path = Config.FAISS_INDEX_PATH.replace('.faiss', '_recipes.pkl')
                if os.path.exists(recipes_pickle_path):
                    with open(recipes_pickle_path, 'rb') as f:
                        self.recipes_data = pickle.load(f)
                else:
                    logger.warning("Recipe data pickle not found, creating new index")
                    self._create_faiss_index()
            else:
                logger.info("FAISS index not found, creating new one")
                self._create_faiss_index()
                
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            self._create_faiss_index()
    
    def _create_faiss_index(self):
        """Create new FAISS index from recipe data"""
        try:
            # Load recipe dataset
            recipes = self._load_recipe_dataset()
            if not recipes:
                logger.warning("No recipe data available, using fallback recipes")
                recipes = self._get_fallback_recipes()
            
            # Create embeddings
            logger.info("Creating recipe embeddings")
            recipe_texts = self._prepare_recipe_texts(recipes)
            embeddings = self.sentence_transformer.encode(recipe_texts, normalize_embeddings=True)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            self.faiss_index.add(embeddings.astype('float32'))
            
            # Save index and data
            faiss.write_index(self.faiss_index, Config.FAISS_INDEX_PATH)
            
            # Save recipe data
            self.recipes_data = recipes
            recipes_pickle_path = Config.FAISS_INDEX_PATH.replace('.faiss', '_recipes.pkl')
            with open(recipes_pickle_path, 'wb') as f:
                pickle.dump(recipes, f)
            
            logger.info(f"Created FAISS index with {len(recipes)} recipes")
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            self._create_fallback_system()
    
    def _load_recipe_dataset(self) -> List[Dict[str, Any]]:
        """Load recipe dataset from CSV file"""
        try:
            if not os.path.exists(Config.RECIPE_DATASET_PATH):
                logger.warning(f"Recipe dataset not found at {Config.RECIPE_DATASET_PATH}")
                return []
            
            df = pd.read_csv(Config.RECIPE_DATASET_PATH)
            recipes = []
            
            for _, row in df.iterrows():
                recipe = {
                    'title': row.get('title', 'Unknown Recipe'),
                    'ingredients': row.get('ingredients', '').split(',') if pd.notna(row.get('ingredients')) else [],
                    'instructions': row.get('instructions', '').split('.') if pd.notna(row.get('instructions')) else [],
                    'cooking_time': row.get('cooking_time', 30),
                    'difficulty': row.get('difficulty', 'medium'),
                    'cuisine': row.get('cuisine', 'general'),
                    'tags': row.get('tags', '').split(',') if pd.notna(row.get('tags')) else []
                }
                recipes.append(recipe)
            
            logger.info(f"Loaded {len(recipes)} recipes from dataset")
            return recipes
            
        except Exception as e:
            logger.error(f"Error loading recipe dataset: {e}")
            return []
    
    def _get_fallback_recipes(self) -> List[Dict[str, Any]]:
        """Get fallback recipes when dataset is not available"""
        return [
            {
                'title': 'Simple Stir Fry',
                'ingredients': ['vegetables', 'oil', 'soy sauce', 'garlic', 'onion'],
                'instructions': ['Heat oil in a pan', 'Add chopped vegetables', 'Stir fry for 5-7 minutes', 'Add soy sauce and serve'],
                'cooking_time': 15,
                'difficulty': 'easy',
                'cuisine': 'asian',
                'tags': ['quick', 'vegetarian', 'healthy']
            },
            {
                'title': 'Quick Salad',
                'ingredients': ['lettuce', 'tomato', 'cucumber', 'olive oil', 'salt', 'pepper'],
                'instructions': ['Wash and chop vegetables', 'Mix in a bowl', 'Drizzle with olive oil', 'Season with salt and pepper'],
                'cooking_time': 10,
                'difficulty': 'easy',
                'cuisine': 'mediterranean',
                'tags': ['fresh', 'vegetarian', 'healthy']
            },
            {
                'title': 'Simple Pasta',
                'ingredients': ['pasta', 'tomato', 'garlic', 'olive oil', 'basil'],
                'instructions': ['Boil pasta according to package instructions', 'Sauté garlic in olive oil', 'Add chopped tomatoes', 'Mix with cooked pasta'],
                'cooking_time': 20,
                'difficulty': 'easy',
                'cuisine': 'italian',
                'tags': ['quick', 'vegetarian']
            },
            {
                'title': 'Quick Omelette',
                'ingredients': ['eggs', 'cheese', 'vegetables', 'butter', 'salt', 'pepper'],
                'instructions': ['Beat eggs in a bowl', 'Heat butter in a pan', 'Pour eggs and add cheese/vegetables', 'Fold and cook until done'],
                'cooking_time': 10,
                'difficulty': 'easy',
                'cuisine': 'french',
                'tags': ['breakfast', 'protein']
            },
            {
                'title': 'Simple Rice Bowl',
                'ingredients': ['rice', 'vegetables', 'soy sauce', 'sesame oil', 'garlic'],
                'instructions': ['Cook rice according to package instructions', 'Steam or sauté vegetables', 'Combine rice and vegetables', 'Add soy sauce and serve'],
                'cooking_time': 25,
                'difficulty': 'easy',
                'cuisine': 'asian',
                'tags': ['vegetarian', 'healthy']
            }
        ]
    
    def _prepare_recipe_texts(self, recipes: List[Dict[str, Any]]) -> List[str]:
        """Prepare recipe texts for embedding"""
        texts = []
        for recipe in recipes:
            # Combine title, ingredients, and tags for search
            ingredients_text = ' '.join(recipe.get('ingredients', []))
            tags_text = ' '.join(recipe.get('tags', []))
            text = f"{recipe.get('title', '')} {ingredients_text} {tags_text}".lower()
            texts.append(text)
        return texts
    
    def _create_fallback_system(self):
        """Create a simple fallback system when RAG fails"""
        logger.warning("Creating fallback recipe system")
        self.recipes_data = self._get_fallback_recipes()
        self.faiss_index = None
    
    def search_recipes(self, pantry_items: List[str], expiring_items: List[str] = None, 
                      max_recipes: int = None) -> List[Dict[str, Any]]:
        """Search for recipes based on pantry items"""
        if not Config.RAG_ENABLE or not self.faiss_index or not self.recipes_data:
            logger.info("RAG disabled or not available, using fallback recipes")
            return self._get_fallback_recipes()[:max_recipes or Config.MAX_RECIPES]
        
        try:
            # Create search query
            query_text = self._create_search_query(pantry_items, expiring_items)
            
            # Encode query
            query_embedding = self.sentence_transformer.encode([query_text], normalize_embeddings=True)
            
            # Search FAISS index
            max_k = max_recipes or Config.MAX_RECIPES
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), max_k)
            
            # Filter by similarity threshold
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= Config.RAG_SIMILARITY_THRESHOLD and idx < len(self.recipes_data):
                    recipe = self.recipes_data[idx].copy()
                    recipe['similarity_score'] = float(score)
                    results.append(recipe)
            
            # Sort by similarity score and prioritize expiring items
            results = self._prioritize_recipes(results, expiring_items or [])
            
            logger.info(f"Found {len(results)} recipes with RAG search")
            return results[:max_recipes or Config.MAX_RECIPES]
            
        except Exception as e:
            logger.error(f"Error in RAG search: {e}")
            return self._get_fallback_recipes()[:max_recipes or Config.MAX_RECIPES]
    
    def _create_search_query(self, pantry_items: List[str], expiring_items: List[str] = None) -> str:
        """Create search query from pantry items"""
        query_parts = []
        
        # Add all pantry items
        query_parts.extend(pantry_items)
        
        # Emphasize expiring items
        if expiring_items:
            # Repeat expiring items to increase their weight
            query_parts.extend(expiring_items * 2)
        
        return ' '.join(query_parts).lower()
    
    def _prioritize_recipes(self, recipes: List[Dict[str, Any]], expiring_items: List[str]) -> List[Dict[str, Any]]:
        """Prioritize recipes that use expiring items"""
        if not expiring_items:
            return sorted(recipes, key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        expiring_set = set(item.lower() for item in expiring_items)
        
        def recipe_score(recipe):
            base_score = recipe.get('similarity_score', 0)
            
            # Check how many expiring items the recipe uses
            ingredients = [ing.lower() for ing in recipe.get('ingredients', [])]
            expiring_matches = sum(1 for item in expiring_set if any(item in ing for ing in ingredients))
            
            # Boost score for recipes using expiring items
            priority_boost = expiring_matches * 0.2
            
            return base_score + priority_boost
        
        return sorted(recipes, key=recipe_score, reverse=True)
    
    def get_recipe_suggestions(self, pantry_items: List[str], expiring_items: List[str] = None,
                             max_recipes: int = None) -> List[Dict[str, Any]]:
        """Get recipe suggestions with RAG enhancement"""
        # Get base recipes from RAG
        recipes = self.search_recipes(pantry_items, expiring_items, max_recipes)
        
        # Enhance recipes with additional information
        enhanced_recipes = []
        for recipe in recipes:
            enhanced_recipe = recipe.copy()
            
            # Add metadata
            enhanced_recipe['source'] = 'rag_system'
            enhanced_recipe['pantry_items_used'] = self._get_used_pantry_items(recipe, pantry_items)
            enhanced_recipe['expiring_items_used'] = self._get_used_pantry_items(recipe, expiring_items or [])
            
            enhanced_recipes.append(enhanced_recipe)
        
        return enhanced_recipes
    
    def _get_used_pantry_items(self, recipe: Dict[str, Any], pantry_items: List[str]) -> List[str]:
        """Get pantry items used in the recipe"""
        recipe_ingredients = [ing.lower() for ing in recipe.get('ingredients', [])]
        used_items = []
        
        for item in pantry_items:
            item_lower = item.lower()
            if any(item_lower in ing for ing in recipe_ingredients):
                used_items.append(item)
        
        return used_items
    
    def update_recipe_index(self):
        """Update the recipe index with new data"""
        logger.info("Updating recipe index")
        self._create_faiss_index()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get RAG system status"""
        return {
            'rag_enabled': Config.RAG_ENABLE,
            'sentence_transformer_loaded': self.sentence_transformer is not None,
            'faiss_index_loaded': self.faiss_index is not None,
            'recipes_count': len(self.recipes_data) if self.recipes_data else 0,
            'faiss_index_path': Config.FAISS_INDEX_PATH,
            'recipe_dataset_path': Config.RECIPE_DATASET_PATH
        }
