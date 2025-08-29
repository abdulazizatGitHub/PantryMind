#!/usr/bin/env python3
"""
RecipeNLG RAG System Setup
Builds vector store and metadata for recipe retrieval using FAISS and sentence transformers
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional

# Add the app directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent / "app"))

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError as e:
    print(f"❌ Required packages not installed: {e}")
    print("Please install: pip install sentence-transformers faiss-cpu")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RecipeNLGRAGSetup:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the RAG setup with sentence transformer model"""
        self.model_name = model_name
        self.embedder = None
        self.recipes_data = []
        self.recipe_texts = []
        self.embeddings = None
        self.faiss_index = None
        
    def load_sentence_transformer(self):
        """Load the sentence transformer model"""
        logger.info(f"Loading sentence transformer model: {self.model_name}")
        try:
            self.embedder = SentenceTransformer(self.model_name)
            logger.info("✅ Sentence transformer loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load sentence transformer: {e}")
            return False
    
    def load_recipenlg_dataset(self, csv_path: Path, max_recipes: Optional[int] = None) -> bool:
        """Load and preprocess RecipeNLG dataset"""
        logger.info(f"Loading RecipeNLG dataset from {csv_path}")
        
        if not csv_path.exists():
            logger.error(f"Dataset file not found: {csv_path}")
            return False
        
        try:
            # Read CSV in chunks to handle large files
            chunk_size = 10000
            total_loaded = 0
            
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                logger.info(f"Processing chunk {total_loaded//chunk_size + 1}")
                
                for _, row in chunk.iterrows():
                    if max_recipes and total_loaded >= max_recipes:
                        break
                    
                    # Extract and clean data
                    recipe = self._process_recipe_row(row)
                    if recipe:
                        self.recipes_data.append(recipe)
                        total_loaded += 1
                
                if max_recipes and total_loaded >= max_recipes:
                    break
            
            logger.info(f"✅ Loaded {len(self.recipes_data)} recipes from dataset")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
            return False
    
    def _process_recipe_row(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Process a single recipe row from the dataset"""
        try:
            title = row['title'] if pd.notna(row['title']) else "Unknown Recipe"
            ingredients = self._clean_ingredients(row['ingredients'])
            instructions = self._clean_instructions(row['directions'])
            
            # Skip if no valid ingredients or instructions
            if not ingredients or not instructions:
                return None
            
            # Estimate cooking time and servings
            prep_time, cook_time = self._estimate_cooking_time(ingredients, instructions)
            servings = self._estimate_servings(ingredients, instructions)
            
            recipe = {
                'title': title,
                'ingredients': ingredients,
                'instructions': instructions,
                'prep_time_minutes': prep_time,
                'cook_time_minutes': cook_time,
                'servings': servings,
                'source': row.get('source', 'RecipeNLG'),
                'link': row.get('link', ''),
                'ner_tags': self._extract_ner_tags(row.get('NER', ''))
            }
            
            return recipe
            
        except Exception as e:
            logger.warning(f"Failed to process recipe row: {e}")
            return None
    
    def _clean_ingredients(self, ingredients_str: str) -> List[str]:
        """Clean and parse ingredients string into a list"""
        if pd.isna(ingredients_str) or not ingredients_str:
            return []
        
        # Remove quotes and brackets, split by comma
        ingredients = ingredients_str.strip('[]"').split('", "')
        cleaned = []
        
        for ing in ingredients:
            ing = ing.strip().strip('"').strip()
            if ing:
                # Remove common measurement prefixes
                import re
                ing = re.sub(r'^\d+\s*(c\.|cup|cups|Tbsp\.|tbsp\.|tsp\.|teaspoon|teaspoons|tablespoon|tablespoons|oz\.|ounce|ounces|pound|pounds|lb\.|lbs\.|g|gram|grams|kg|kilogram|kilograms)\s*', '', ing, flags=re.IGNORECASE)
                ing = ing.strip()
                if ing:
                    cleaned.append(ing)
        
        return cleaned
    
    def _clean_instructions(self, instructions_str: str) -> List[str]:
        """Clean and parse instructions string into a list of steps"""
        if pd.isna(instructions_str) or not instructions_str:
            return []
        
        # Remove quotes and brackets, split by comma
        instructions = instructions_str.strip('[]"').split('", "')
        cleaned = []
        
        for inst in instructions:
            inst = inst.strip().strip('"').strip()
            if inst:
                # Remove extra whitespace and normalize
                import re
                inst = re.sub(r'\s+', ' ', inst)
                if inst:
                    cleaned.append(inst)
        
        return cleaned
    
    def _estimate_cooking_time(self, ingredients: List[str], instructions: List[str]) -> tuple:
        """Estimate cooking time based on ingredients and instructions"""
        base_prep = 10
        base_cook = 20
        
        # Adjust based on number of ingredients
        if len(ingredients) > 8:
            base_prep += 10
        elif len(ingredients) > 5:
            base_prep += 5
        
        # Adjust based on cooking methods mentioned
        cooking_methods = ['bake', 'roast', 'slow cook', 'simmer', 'boil', 'grill']
        for method in cooking_methods:
            if any(method in inst.lower() for inst in instructions):
                if method in ['bake', 'roast', 'slow cook']:
                    base_cook += 30
                elif method in ['simmer', 'boil']:
                    base_cook += 15
        
        return base_prep, base_cook
    
    def _estimate_servings(self, ingredients: List[str], instructions: List[str]) -> int:
        """Estimate number of servings based on ingredients and instructions"""
        base_servings = 4
        
        # Adjust based on protein amounts
        protein_keywords = ['chicken breast', 'beef', 'pork', 'fish', 'steak']
        for keyword in protein_keywords:
            if any(keyword in ing.lower() for ing in ingredients):
                if any('whole chicken' in ing.lower() for ing in ingredients):
                    base_servings = 6
                elif any('breast' in ing.lower() for ing in ingredients):
                    base_servings = 2
                break
        
        # Adjust based on pasta/rice amounts
        if any('pasta' in ing.lower() or 'rice' in ing.lower() for ing in ingredients):
            base_servings = 6
        
        return max(1, min(8, base_servings))
    
    def _extract_ner_tags(self, ner_str: str) -> List[str]:
        """Extract NER tags from the NER column"""
        if pd.isna(ner_str) or not ner_str:
            return []
        
        try:
            # Remove quotes and brackets, split by comma
            tags = ner_str.strip('[]"').split('", "')
            cleaned_tags = [tag.strip().strip('"').strip() for tag in tags if tag.strip()]
            return cleaned_tags
        except:
            return []
    
    def prepare_recipe_texts(self):
        """Prepare recipe texts for embedding"""
        logger.info("Preparing recipe texts for embedding")
        
        self.recipe_texts = []
        for recipe in self.recipes_data:
            # Combine title, ingredients, and NER tags for search
            ingredients_text = ' '.join(recipe.get('ingredients', []))
            ner_text = ' '.join(recipe.get('ner_tags', []))
            
            # Create searchable text
            search_text = f"{recipe.get('title', '')} {ingredients_text} {ner_text}".lower()
            self.recipe_texts.append(search_text)
        
        logger.info(f"✅ Prepared {len(self.recipe_texts)} recipe texts")
    
    def create_embeddings(self):
        """Create embeddings for all recipe texts"""
        if not self.embedder:
            logger.error("Sentence transformer not loaded")
            return False
        
        if not self.recipe_texts:
            logger.error("No recipe texts prepared")
            return False
        
        logger.info("Creating embeddings...")
        try:
            self.embeddings = self.embedder.encode(
                self.recipe_texts, 
                show_progress_bar=True, 
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            logger.info(f"✅ Created embeddings with shape: {self.embeddings.shape}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create embeddings: {e}")
            return False
    
    def build_faiss_index(self):
        """Build FAISS index from embeddings"""
        if self.embeddings is None:
            logger.error("No embeddings available")
            return False
        
        logger.info("Building FAISS index...")
        try:
            dimension = self.embeddings.shape[1]
            
            # Use IndexFlatIP for cosine similarity (since embeddings are normalized)
            self.faiss_index = faiss.IndexFlatIP(dimension)
            
            # Add embeddings to index
            self.faiss_index.add(self.embeddings.astype('float32'))
            
            logger.info(f"✅ FAISS index built with {self.faiss_index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to build FAISS index: {e}")
            return False
    
    def save_rag_system(self, output_dir: Path):
        """Save the complete RAG system"""
        logger.info(f"Saving RAG system to {output_dir}")
        
        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss_path = output_dir / "recipe_index.faiss"
            faiss.write_index(self.faiss_index, str(faiss_path))
            logger.info(f"✅ FAISS index saved to {faiss_path}")
            
            # Save recipe metadata
            metadata_path = output_dir / "recipes_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.recipes_data, f)
            logger.info(f"✅ Recipe metadata saved to {metadata_path}")
            
            # Save recipe texts
            texts_path = output_dir / "recipe_texts.pkl"
            with open(texts_path, 'wb') as f:
                pickle.dump(self.recipe_texts, f)
            logger.info(f"✅ Recipe texts saved to {texts_path}")
            
            # Save system info
            info = {
                'model_name': self.model_name,
                'total_recipes': len(self.recipes_data),
                'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else None,
                'faiss_index_type': 'IndexFlatIP',
                'created_at': pd.Timestamp.now().isoformat()
            }
            
            info_path = output_dir / "system_info.json"
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
            logger.info(f"✅ System info saved to {info_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save RAG system: {e}")
            return False
    
    def test_rag_system(self, test_queries: List[str] = None) -> bool:
        """Test the RAG system with sample queries"""
        if not self.faiss_index or not self.recipes_data:
            logger.error("RAG system not built")
            return False
        
        logger.info("Testing RAG system...")
        
        if test_queries is None:
            test_queries = [
                "chicken rice tomatoes",
                "pasta cheese vegetables",
                "beef steak potatoes",
                "fish lemon herbs"
            ]
        
        try:
            for query in test_queries:
                logger.info(f"\n🔍 Testing query: '{query}'")
                
                # Encode query
                query_embedding = self.embedder.encode([query], normalize_embeddings=True)
                
                # Search
                scores, indices = self.faiss_index.search(query_embedding.astype('float32'), 3)
                
                # Display results
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx < len(self.recipes_data):
                        recipe = self.recipes_data[idx]
                        logger.info(f"  {i+1}. {recipe['title']} (score: {score:.3f})")
                        logger.info(f"     Ingredients: {', '.join(recipe['ingredients'][:3])}...")
            
            logger.info("✅ RAG system test completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ RAG system test failed: {e}")
            return False

def main():
    """Main function to set up RecipeNLG RAG system"""
    logger.info("🚀 RecipeNLG RAG System Setup")
    logger.info("=" * 50)
    
    # Configuration
    csv_path = Path("data/RecipeNLG/RecipeNLG_dataset.csv")
    output_dir = Path("data/faiss")
    max_recipes = 100000  # Limit for initial setup
    
    # Check if dataset exists
    if not csv_path.exists():
        logger.error(f"RecipeNLG dataset not found at {csv_path}")
        logger.info("Please ensure the RecipeNLG_dataset.csv file is in the data/RecipeNLG/ directory")
        return False
    
    try:
        # Initialize RAG setup
        rag_setup = RecipeNLGRAGSetup()
        
        # Step 1: Load sentence transformer
        if not rag_setup.load_sentence_transformer():
            return False
        
        # Step 2: Load dataset
        if not rag_setup.load_recipenlg_dataset(csv_path, max_recipes):
            return False
        
        # Step 3: Prepare recipe texts
        rag_setup.prepare_recipe_texts()
        
        # Step 4: Create embeddings
        if not rag_setup.create_embeddings():
            return False
        
        # Step 5: Build FAISS index
        if not rag_setup.build_faiss_index():
            return False
        
        # Step 6: Save RAG system
        if not rag_setup.save_rag_system(output_dir):
            return False
        
        # Step 7: Test RAG system
        rag_setup.test_rag_system()
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 RecipeNLG RAG System Setup Complete!")
        logger.info("=" * 60)
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"📊 Total recipes: {len(rag_setup.recipes_data)}")
        logger.info(f"🔢 Embedding dimension: {rag_setup.embeddings.shape[1]}")
        logger.info(f"📚 FAISS index: {rag_setup.faiss_index.ntotal} vectors")
        
        logger.info("\n📋 Next steps:")
        logger.info("1. Update your config.py to use the new FAISS index:")
        logger.info(f"   FAISS_INDEX_PATH = '{output_dir}/recipe_index.faiss'")
        logger.info("2. Restart your application to use the new RAG system")
        logger.info("3. Test recipe retrieval with your API endpoints")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
