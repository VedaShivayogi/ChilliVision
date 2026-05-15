@echo off
title ChilliVision — Explainable Chilli Quality Grading
echo.
echo  🌶  ChilliVision — Explainable Chilli Quality Grading
echo  ==================================================
echo.

echo  Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found. Install Python 3.9+ from python.org
    pause
    exit /b 1
)

echo  Installing dependencies...
pip install -r requirements.txt -q

echo  Starting Flask server...
echo.
echo  Web UI : http://localhost:5000
echo  API    : http://localhost:5000/api/analyse
echo.

cd backend
python app.py
pause
