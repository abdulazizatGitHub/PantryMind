# 🍽️ PantryMind - AI-Powered Recipe Generator

An intelligent web application that helps households reduce food waste by scanning pantry items, detecting ingredients using AI, and generating personalized recipes using advanced RAG (Retrieval-Augmented Generation) technology.

## 🌟 Key Features

- **📷 AI-Powered Ingredient Detection**: Uses YOLOv8 to detect food items from camera images
- **🧠 Advanced RAG System**: Retrieval-Augmented Generation for intelligent recipe suggestions
- **📊 Real-Time Analytics Dashboard**: Track waste reduction progress and environmental impact
- **🎯 Smart Recipe Generation**: AI suggests recipes using available ingredients with waste reduction focus
- **📱 Modern Web Interface**: Professional React frontend with responsive design
- **🔄 Real-Time Updates**: Dashboard analytics update automatically after recipe generation

## 🏗️ System Architecture

### Backend (FastAPI + AI Models)
```
backend/
├── app/
│   ├── ai/                    # AI model implementations
│   │   ├── yolov8_infer.py   # YOLOv8 inference
│   │   ├── ocr_util.py       # OCR utilities
│   │   ├── rag_prompt.py     # RAG prompts
│   │   └── retrieval_index.py # FAISS index management
│   ├── routers/              # API endpoints
│   │   ├── detect.py         # Ingredient detection
│   │   ├── recipes.py        # Recipe generation
│   │   ├── pantry.py         # Pantry management
│   │   └── ocr.py           # OCR processing
│   ├── rag_system.py         # RAG system implementation
│   ├── ai_models.py          # AI model classes
│   └── config.py             # Configuration
├── data/
│   ├── RecipeNLG/            # RecipeNLG dataset
│   ├── faiss/               # FAISS index storage
│   └── results/             # JSON fallback storage
└── scripts/                 # Setup and utility scripts
```

### Frontend (React + Vite)
```
frontend/
├── src/
│   ├── components/
│   │   ├── Dashboard.jsx     # Analytics dashboard
│   │   ├── Scanner.jsx       # Ingredient scanning
│   │   ├── RecipeView.jsx    # Recipe display
│   │   ├── RecipeGenerator.jsx # Recipe generation
│   │   ├── Navigation.jsx    # Navigation bar
│   │   └── Pantry.jsx        # Pantry management
│   └── App.jsx              # Main application
```

## 🤖 AI Models & Technologies

### 1. **YOLOv8 (Object Detection)**
- **Model**: `yolov8n.pt` (nano) - 6.2MB, optimized for speed
- **Purpose**: Detect food ingredients in uploaded images
- **Features**:
  - Real-time inference
  - Configurable confidence threshold (default: 0.3)
  - Automatic model download if not present
  - GPU acceleration support

### 2. **RAG System (Recipe Generation)**
- **Sentence Transformer**: `all-MiniLM-L6-v2`
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Dataset**: RecipeNLG (2.1GB, 2.3M+ recipes)
- **Features**:
  - Semantic recipe search
  - Ingredient-based filtering
  - Top-K retrieval (default: 5 recipes)
  - Similarity threshold filtering

### 3. **EasyOCR (Text Recognition)**
- **Purpose**: Extract expiry dates from packaging
- **Languages**: English (configurable)
- **Features**: GPU acceleration support

### 4. **OpenAI Integration (Optional)**
- **Model**: GPT-3.5-turbo
- **Purpose**: Enhanced recipe generation and rewriting
- **Features**: Fallback to RAG system if API key not available

## 📊 Dataset: RecipeNLG

### **Source**: [RecipeNLG Dataset](https://github.com/Glorf/recipenlg)
- **Size**: 2.1GB CSV file
- **Records**: 2.3M+ recipes
- **Format**: CSV with recipe metadata
- **Processing**: Converted to FAISS index for fast retrieval

### **Data Structure**:
```csv
title,ingredients,instructions,servings,cooking_time,difficulty,cuisine,tags
```

### **Setup Process**:
1. Download RecipeNLG dataset
2. Preprocess for RAG system
3. Create FAISS index with sentence embeddings
4. Store metadata for recipe retrieval

## 🔄 System Flow

### **1. Ingredient Detection Flow**
```
User Upload Image → YOLOv8 Detection → Ingredient Grouping → Recipe Generation → Analytics Update
```

### **2. Recipe Generation Flow**
```
Detected Ingredients → RAG System Query → FAISS Retrieval → Recipe Enhancement → Dashboard Update
```

### **3. Analytics Flow**
```
Recipe Generation → Waste Reduction Calculation → CO₂ Impact → Money Savings → Real-time Dashboard Update
```

## 🛠️ Tech Stack

### **Backend**
- **Framework**: FastAPI (Python)
- **Database**: MongoDB (with JSON fallback)
- **AI Models**: 
  - YOLOv8 (Ultralytics)
  - Sentence Transformers
  - FAISS (Facebook AI)
  - EasyOCR
- **Dependencies**: PyTorch, OpenCV, Pandas, NumPy

### **Frontend**
- **Framework**: React 18
- **Build Tool**: Vite
- **Styling**: Custom CSS + TailwindCSS
- **HTTP Client**: Axios
- **Routing**: React Router DOM

### **AI/ML Libraries**
- **torch==2.6.0**: PyTorch for deep learning
- **ultralytics==8.0.196**: YOLOv8 implementation
- **sentence-transformers==2.2.2**: Text embeddings
- **faiss-cpu**: Vector similarity search
- **easyocr==1.7.0**: Text recognition

## 🚀 Quick Start

### **Prerequisites**
- Python 3.8+
- Node.js 16+
- MongoDB (local or cloud)
- OpenAI API key (optional)

### **1. Clone Repository**
```bash
git clone <repository-url>
cd PantryMind
```

### **2. Backend Setup**
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up AI models and RAG system
python scripts/setup_recipenlg_complete.py

# Configure environment
cp env.example .env
# Edit .env with your settings
```

### **3. Frontend Setup**
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### **4. Start System**
```bash
# From project root
./start.sh  # Linux/Mac
# OR
start.bat   # Windows
```

## ⚙️ Configuration

### **Environment Variables** (`backend/.env`)
```env
# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017/food_waste_reducer
MONGO_DB_NAME=food_waste_reducer

# AI Model Configuration
YOLO_MODEL_PATH=yolov8n.pt
YOLO_CONFIDENCE_THRESHOLD=0.3
YOLO_IMAGE_SIZE=640

# RAG System Configuration
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
FAISS_INDEX_PATH=data/faiss/recipe_index.faiss
RECIPE_DATASET_PATH=data/faiss/recipes_metadata.pkl
RAG_TOP_K=5
RAG_SIMILARITY_THRESHOLD=0.3

# OpenAI Configuration (Optional)
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=1000
OPENAI_TEMPERATURE=0.3

# EasyOCR Configuration
EASYOCR_LANGUAGES=en
EASYOCR_GPU=False
```

## 📡 API Endpoints

### **Core Detection**
- `POST /api/detect` - Basic ingredient detection
- `POST /api/scan-and-generate-recipes` - Complete scan + recipe generation

### **Recipe Management**
- `POST /api/recipes/suggest` - Generate recipes from pantry
- `GET /api/recipes/recipes` - Get all recipes
- `POST /api/recipes/{recipe_id}/use` - Mark recipe as used

### **Analytics**
- `GET /api/analytics/waste-reduction` - Waste reduction statistics
- `GET /api/analytics/recipe-stats` - Recipe generation statistics

### **Pantry Management**
- `GET /api/pantry/items` - Get pantry items
- `POST /api/pantry/items` - Add pantry item
- `DELETE /api/pantry/items/{id}` - Remove pantry item

## 🎯 Key Features Explained

### **1. Smart Ingredient Grouping**
- Automatically groups duplicate detections (e.g., 3 oranges → 1 card with count)
- Calculates average confidence across detections
- Shows detection count and max confidence

### **2. Real-Time Analytics**
- Dashboard updates automatically after recipe generation
- Tracks waste reduction, CO₂ savings, money saved
- Shows last updated timestamp and refresh controls

### **3. Professional UI/UX**
- Modern gradient designs with glassmorphism effects
- Responsive layout for all devices
- Smooth animations and hover effects
- Professional color scheme and typography

### **4. Fallback Systems**
- JSON file storage if MongoDB unavailable
- RAG system fallback if OpenAI API fails
- Graceful error handling throughout

## 📊 Performance Metrics

### **Model Performance**
- **YOLOv8**: ~30ms inference time (CPU)
- **RAG System**: ~100ms recipe retrieval
- **FAISS Index**: Sub-second similarity search

### **System Capabilities**
- **Recipe Database**: 2.3M+ recipes
- **Ingredient Detection**: 80+ food categories
- **Real-time Processing**: <2 seconds end-to-end
- **Scalability**: Handles multiple concurrent users

## 🔧 Development

### **Running Tests**
```bash
cd backend
python test_models.py
python test_rag.py
```

### **Model Training** (Optional)
```bash
cd backend
python scripts/train_yolo.py  # Custom YOLO training
```

### **RAG System Management**
```bash
cd backend
python scripts/setup_recipenlg_complete.py  # Full setup
python force_reload_rag.py                  # Force reload
```

## 🚀 Deployment

### **Backend Deployment**
```bash
# Production server
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker

# Environment variables
export MONGO_URI=your_mongodb_uri
export OPENAI_API_KEY=your_openai_key
```

### **Frontend Deployment**
```bash
cd frontend
npm run build
# Deploy dist/ folder to your hosting service
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **RecipeNLG Dataset**: [Glorf/recipenlg](https://github.com/Glorf/recipenlg)
- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Sentence Transformers**: [Hugging Face](https://huggingface.co/sentence-transformers)
- **FAISS**: [Facebook Research](https://github.com/facebookresearch/faiss)

## 📞 Support

For questions or issues, please open a GitHub issue or contact the development team.

---

**Made with ❤️ to reduce food waste and save the planet! 🌱**

*PantryMind - Where AI meets sustainability*
