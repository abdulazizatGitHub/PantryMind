# 🍽️ AI Food Waste Reducer

An intelligent web application that helps households reduce food waste by scanning pantry items, detecting expiry dates, and suggesting recipes based on available ingredients.

## 🌟 Features

- **📷 AI-Powered Food Detection**: Uses YOLOv8 to detect food items from camera images
- **📅 Expiry Date Recognition**: EasyOCR extracts expiry dates from packaging
- **👨‍🍳 Smart Recipe Generation**: AI suggests recipes using available pantry items
- **📊 Waste Tracking Dashboard**: Monitor your food waste reduction progress
- **📱 Modern Web Interface**: Responsive design with TailwindCSS

## 🛠️ Tech Stack

### Backend
- **Python Flask**: RESTful API server
- **MongoDB**: NoSQL database for data persistence
- **YOLOv8**: Object detection for food items
- **EasyOCR**: Text recognition for expiry dates
- **OpenAI API**: Recipe generation (optional)

### Frontend
- **React 18**: Modern UI framework
- **Vite**: Fast build tool
- **TailwindCSS**: Utility-first CSS framework
- **Webcam API**: Camera integration for scanning

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- MongoDB (local or cloud)
- OpenAI API key (optional, for enhanced recipe generation)

### Option 1: Automated Setup (Recommended)

1. **Run the automated setup script**:
   ```bash
   cd backend
   python setup_models.py
   ```
   
   This will automatically:
   - Install all dependencies
   - Download YOLOv8 model
   - Set up EasyOCR
   - Configure OpenAI (if API key provided)
   - Create environment configuration

2. **Start the application**:
   ```bash
   # From project root
   ./start.sh  # Linux/Mac
   # OR
   start.bat   # Windows
   ```

### Option 2: Manual Setup

#### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up AI models**:
   ```bash
   python setup_models.py
   ```

5. **Configure environment variables**:
   Edit the `.env` file in the backend directory:
   ```env
   # MongoDB Configuration
   MONGO_URI=mongodb://localhost:27017/food_waste_reducer
   
   # OpenAI API Configuration (optional)
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-3.5-turbo
   
   # Flask Configuration
   SECRET_KEY=your_secret_key_here
   
   # AI Model Configuration
   YOLO_MODEL_PATH=yolov8n.pt
   YOLO_CONFIDENCE_THRESHOLD=0.3
   EASYOCR_LANGUAGES=en
   ```

6. **Test the models**:
   ```bash
   python test_models.py
   ```

7. **Start MongoDB** (if running locally):
   ```bash
   mongod
   ```

8. **Run the Flask server**:
   ```bash
   python run.py
   ```
   The backend will be available at `http://localhost:5000`

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```
   The frontend will be available at `http://localhost:5173`

## 📖 Usage Guide

### 1. Dashboard
- View your pantry statistics
- Track waste reduction progress
- See recent activity

### 2. Pantry Management
- Add items manually with name, quantity, and expiry date
- View all items with expiry status indicators
- Delete items when used or expired

### 3. Food Scanner
- Use your device camera to scan food items
- AI automatically detects items and extracts expiry dates
- Items are automatically added to your pantry

### 4. Recipe Generator
- Generate recipes based on all pantry items
- Select specific items for targeted recipes
- Mark recipes as used to track waste reduction

## 🔧 Configuration

### AI Models
- **YOLOv8**: Pre-trained model for food detection
  - Model options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large), yolov8x.pt (extra large)
  - Configurable confidence threshold and image size
  - Automatic GPU acceleration if available
- **EasyOCR**: Configured for English text recognition
  - Support for multiple languages
  - GPU acceleration option
  - Optimized for expiry date extraction
- **Recipe Generation**: Uses OpenAI GPT-3.5-turbo (fallback to templates if no API key)
  - Configurable model, tokens, and temperature
  - Cost-effective recipe generation
  - Fallback templates for offline use

### Model Setup
For detailed model configuration and setup instructions, see [MODEL_SETUP.md](MODEL_SETUP.md).

### Database Schema
- **pantry_items**: Food items with expiry dates
- **recipes**: Generated and saved recipes
- **user_interactions**: Activity tracking for analytics

## 🚀 Deployment

### Backend (Render/Railway)
1. Connect your GitHub repository
2. Set environment variables
3. Deploy with Python runtime

### Frontend (Vercel)
1. Connect your GitHub repository
2. Set build command: `npm run build`
3. Set output directory: `dist`

## 📊 API Endpoints

### Main Routes
- `GET /health` - Health check
- `GET /dashboard` - Dashboard statistics
- `POST /interaction` - Log user interaction

### Pantry Management
- `GET /api/pantry/items` - Get all pantry items
- `POST /api/pantry/items` - Add new item
- `PUT /api/pantry/items/<id>` - Update item
- `DELETE /api/pantry/items/<id>` - Delete item
- `GET /api/pantry/items/expiring` - Get expiring items

### Detection
- `POST /api/detection/scan` - Scan image for food items
- `POST /api/detection/scan-and-add` - Scan and add to pantry
- `POST /api/detection/ocr` - Extract text from image

### Recipes
- `GET /api/recipes/suggest-from-pantry` - Generate from all items
- `POST /api/recipes/suggest` - Generate from selected items
- `GET /api/recipes/recipes` - Get all recipes
- `POST /api/recipes/recipes/<id>/use` - Mark recipe as used

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics for object detection
- EasyOCR for text recognition
- OpenAI for recipe generation capabilities
- MongoDB for data persistence
- React and Vite for the frontend framework

## 📞 Support

For questions or issues, please open a GitHub issue or contact the development team.

---

**Made with ❤️ to reduce food waste and save the planet! 🌱**
