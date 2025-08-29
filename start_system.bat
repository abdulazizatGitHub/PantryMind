@echo off
echo Starting PantryMind AI System...
echo.

echo Starting Backend Server (FastAPI)...
start "Backend Server" cmd /k "cd backend && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak > nul

echo Starting Frontend Server (React)...
start "Frontend Server" cmd /k "cd frontend && npm run dev"

echo.
echo System starting up...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Press any key to close this window...
pause > nul
