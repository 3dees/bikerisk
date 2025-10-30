#!/bin/bash

# Start script for E-Bike Standards Extractor
# Runs both FastAPI backend and Streamlit frontend

echo "🚴 E-Bike Standards Requirement Extractor"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.10+."
    exit 1
fi

# Check if dependencies are installed
if ! python -c "import fastapi" &> /dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    echo ""
fi

# Start FastAPI in background
echo "🚀 Starting FastAPI backend on http://localhost:8000"
python main.py &
FASTAPI_PID=$!

# Wait for FastAPI to start
sleep 3

# Start Streamlit
echo "🌐 Starting Streamlit UI on http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""
streamlit run app.py

# Cleanup: Kill FastAPI when Streamlit exits
echo ""
echo "🛑 Stopping servers..."
kill $FASTAPI_PID
echo "✅ Servers stopped"
