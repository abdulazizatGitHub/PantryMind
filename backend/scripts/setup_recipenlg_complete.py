#!/usr/bin/env python3
"""
Complete RecipeNLG Setup Script
Handles both OpenAI fine-tuning data preparation and RAG system setup
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_prerequisites():
    """Check if all required packages are installed"""
    logger.info("🔍 Checking prerequisites...")
    
    required_packages = [
        'pandas',
        'numpy',
        'sentence_transformers',
        'faiss',
        'openai'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"❌ {package} - not installed")
    
    if missing_packages:
        logger.error(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        logger.info("Please install missing packages:")
        logger.info(f"pip install {' '.join(missing_packages)}")
        return False
    
    logger.info("✅ All required packages are installed")
    return True

def check_openai_cli():
    """Check if OpenAI CLI is installed and configured"""
    logger.info("🔍 Checking OpenAI CLI...")
    
    try:
        result = subprocess.run(['openai', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ OpenAI CLI: {result.stdout.strip()}")
            return True
        else:
            logger.warning("⚠️ OpenAI CLI not working properly")
            return False
    except FileNotFoundError:
        logger.warning("⚠️ OpenAI CLI not found")
        logger.info("Please install OpenAI CLI:")
        logger.info("pip install openai")
        return False

def check_dataset():
    """Check if RecipeNLG dataset exists"""
    logger.info("🔍 Checking RecipeNLG dataset...")
    
    csv_path = Path("data/RecipeNLG/RecipeNLG_dataset.csv")
    
    if not csv_path.exists():
        logger.error(f"❌ RecipeNLG dataset not found at {csv_path}")
        logger.info("Please ensure the RecipeNLG_dataset.csv file is in the data/RecipeNLG/ directory")
        return False
    
    # Check file size
    file_size_mb = csv_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ RecipeNLG dataset found: {file_size_mb:.1f} MB")
    
    return True

def run_finetuning_preparation():
    """Run the fine-tuning data preparation script"""
    logger.info("🚀 Preparing OpenAI fine-tuning data...")
    
    script_path = Path("scripts/prepare_recipenlg_finetuning.py")
    
    if not script_path.exists():
        logger.error(f"❌ Fine-tuning script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Fine-tuning data preparation completed successfully")
            return True
        else:
            logger.error(f"❌ Fine-tuning data preparation failed:")
            logger.error(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running fine-tuning script: {e}")
        return False

def run_rag_setup():
    """Run the RAG system setup script"""
    logger.info("🚀 Setting up RAG system...")
    
    script_path = Path("scripts/setup_recipenlg_rag.py")
    
    if not script_path.exists():
        logger.error(f"❌ RAG setup script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ RAG system setup completed successfully")
            return True
        else:
            logger.error(f"❌ RAG system setup failed:")
            logger.error(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running RAG setup script: {e}")
        return False

def validate_finetuning_data():
    """Validate the generated fine-tuning data"""
    logger.info("🔍 Validating fine-tuning data...")
    
    jsonl_path = Path("data/recipenlg_finetuning.jsonl")
    
    if not jsonl_path.exists():
        logger.error(f"❌ Fine-tuning data not found: {jsonl_path}")
        return False
    
    try:
        # Count lines and validate JSON format
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        valid_entries = 0
        for i, line in enumerate(lines, 1):
            try:
                entry = json.loads(line.strip())
                if 'messages' in entry and len(entry['messages']) == 3:
                    valid_entries += 1
                else:
                    logger.warning(f"⚠️ Invalid entry format at line {i}")
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ JSON decode error at line {i}: {e}")
        
        logger.info(f"✅ Fine-tuning data validation complete:")
        logger.info(f"   Total lines: {len(lines)}")
        logger.info(f"   Valid entries: {valid_entries}")
        
        return valid_entries > 0
        
    except Exception as e:
        logger.error(f"❌ Error validating fine-tuning data: {e}")
        return False

def validate_rag_system():
    """Validate the RAG system setup"""
    logger.info("🔍 Validating RAG system...")
    
    faiss_dir = Path("data/faiss")
    
    if not faiss_dir.exists():
        logger.error(f"❌ FAISS directory not found: {faiss_dir}")
        return False
    
    required_files = [
        "recipe_index.faiss",
        "recipes_metadata.pkl",
        "recipe_texts.pkl",
        "system_info.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not (faiss_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"❌ Missing RAG system files: {', '.join(missing_files)}")
        return False
    
    # Check system info
    try:
        with open(faiss_dir / "system_info.json", 'r') as f:
            info = json.load(f)
        
        logger.info(f"✅ RAG system validation complete:")
        logger.info(f"   Total recipes: {info.get('total_recipes', 'N/A')}")
        logger.info(f"   Embedding dimension: {info.get('embedding_dimension', 'N/A')}")
        logger.info(f"   Model: {info.get('model_name', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error reading RAG system info: {e}")
        return False

def create_finetuning_commands():
    """Create and display fine-tuning commands"""
    logger.info("📋 OpenAI Fine-tuning Commands:")
    logger.info("=" * 50)
    
    jsonl_path = Path("data/recipenlg_finetuning.jsonl")
    
    if not jsonl_path.exists():
        logger.error("❌ Fine-tuning data not found")
        return
    
    logger.info("1. Validate and prepare the data:")
    logger.info(f"   openai tools fine_tunes.prepare_data -f {jsonl_path}")
    logger.info("")
    logger.info("2. Start fine-tuning (development):")
    logger.info("   openai api fine_tunes.create -t recipenlg_prepared.jsonl -m gpt-4o-mini")
    logger.info("")
    logger.info("3. Start fine-tuning (production):")
    logger.info("   openai api fine_tunes.create -t recipenlg_prepared.jsonl -m gpt-4o")
    logger.info("")
    logger.info("4. Monitor fine-tuning progress:")
    logger.info("   openai api fine_tunes.list")
    logger.info("")
    logger.info("5. Get fine-tuning results:")
    logger.info("   openai api fine_tunes.get -i <FINE_TUNE_JOB_ID>")
    logger.info("")
    logger.info("6. Use your fine-tuned model:")
    logger.info("   openai api chat.completions.create -m <FINE_TUNED_MODEL_ID> -g user 'Generate a recipe with chicken and rice'")

def create_config_updates():
    """Create configuration update instructions"""
    logger.info("📋 Configuration Updates:")
    logger.info("=" * 50)
    
    logger.info("1. Update your .env file:")
    logger.info("   # RecipeNLG RAG System")
    logger.info("   FAISS_INDEX_PATH=data/faiss/recipe_index.faiss")
    logger.info("   RECIPE_DATASET_PATH=data/faiss/recipes_metadata.pkl")
    logger.info("   RAG_ENABLE=True")
    logger.info("")
    logger.info("2. Update config.py if needed:")
    logger.info("   FAISS_INDEX_PATH = 'data/faiss/recipe_index.faiss'")
    logger.info("   RECIPE_DATASET_PATH = 'data/faiss/recipes_metadata.pkl'")
    logger.info("")
    logger.info("3. Restart your application to use the new RAG system")

def main():
    """Main setup function"""
    logger.info("🎯 Complete RecipeNLG Setup")
    logger.info("=" * 60)
    logger.info("This script will set up both OpenAI fine-tuning and RAG system")
    logger.info("using the RecipeNLG dataset.")
    
    # Check if running in the correct directory
    if not Path("requirements.txt").exists():
        logger.error("❌ Please run this script from the backend directory")
        return False
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        return False
    
    # Step 2: Check OpenAI CLI
    openai_cli_available = check_openai_cli()
    
    # Step 3: Check dataset
    if not check_dataset():
        return False
    
    # Step 4: Run fine-tuning preparation
    if not run_finetuning_preparation():
        logger.error("❌ Fine-tuning data preparation failed")
        return False
    
    # Step 5: Validate fine-tuning data
    if not validate_finetuning_data():
        logger.error("❌ Fine-tuning data validation failed")
        return False
    
    # Step 6: Run RAG setup
    if not run_rag_setup():
        logger.error("❌ RAG system setup failed")
        return False
    
    # Step 7: Validate RAG system
    if not validate_rag_system():
        logger.error("❌ RAG system validation failed")
        return False
    
    # Success!
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Complete RecipeNLG Setup Successful!")
    logger.info("=" * 60)
    
    # Display next steps
    if openai_cli_available:
        create_finetuning_commands()
    
    create_config_updates()
    
    logger.info("\n📚 What was created:")
    logger.info("✅ Fine-tuning data: data/recipenlg_finetuning.jsonl")
    logger.info("✅ RAG system: data/faiss/")
    logger.info("✅ FAISS index: data/faiss/recipe_index.faiss")
    logger.info("✅ Recipe metadata: data/faiss/recipes_metadata.pkl")
    
    logger.info("\n🚀 You're ready to:")
    logger.info("1. Start OpenAI fine-tuning (if you have API access)")
    logger.info("2. Use the RAG system for recipe retrieval")
    logger.info("3. Integrate with your application")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
