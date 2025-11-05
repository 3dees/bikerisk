#!/bin/bash

echo "🚀 Starting BikeRisk Application..."

# Get the PORT from Railway (Railway sets this automatically)
export STREAMLIT_PORT=${PORT:-8501}
echo "📍 Streamlit will run on port: $STREAMLIT_PORT"

# Start FastAPI backend in background on port 8000
echo "📡 Starting FastAPI backend on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info &

# Store the PID of FastAPI
FASTAPI_PID=$!
echo "FastAPI PID: $FASTAPI_PID"

# Wait for FastAPI to be ready
echo "⏳ Waiting for FastAPI to start..."
sleep 8

# Check if FastAPI is running
if ps -p $FASTAPI_PID > /dev/null; then
   echo "✅ FastAPI is running"
else
   echo "❌ FastAPI failed to start"
   exit 1
fi

# Start Streamlit frontend on Railway's PORT
echo "🎨 Starting Streamlit frontend on port $STREAMLIT_PORT..."
exec streamlit run app.py \
  --server.port $STREAMLIT_PORT \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --browser.gatherUsageStats false