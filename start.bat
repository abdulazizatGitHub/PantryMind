@echo off
echo 🍽️ Starting AI Food Waste Reducer...

REM Check if we're in the right directory
if not exist "backend\requirements.txt" (
    echo ❌ Error: Please run this script from the project root directory
    pause
    exit /b 1
)

REM Start backend
echo 🚀 Starting Flask backend...
cd backend

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install dependencies
echo 📦 Installing Python dependencies...
pip install -r requirements.txt

REM Set up models if not already done
if not exist ".env" (
    echo 🤖 Setting up AI models...
    python setup_models.py
)

REM Start Flask server
echo 🌐 Starting Flask server...
start python run.py

REM Wait a moment for backend to start
timeout /t 5 /nobreak > nul

REM Start frontend
echo 🎨 Starting React frontend...
cd ..\frontend

REM Install dependencies if needed
if not exist "node_modules" (
    echo 📦 Installing Node.js dependencies...
    npm install
)

REM Start development server
start npm run dev

echo ✅ Both servers are starting...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:5173
echo.
echo Press any key to stop...
pause > nul
