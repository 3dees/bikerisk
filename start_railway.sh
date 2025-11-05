#!/bin/bash

echo "🚀 Starting BikeRisk Application..."

# Start FastAPI in background on port 8000
echo "📡 Starting FastAPI backend on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Wait for FastAPI
sleep 3

# Start Streamlit on Railway's PORT (MUST use $PORT, not ${PORT:-8501})
echo "🎨 Starting Streamlit on port $PORT..."
exec streamlit run app.py \
  --server.port $PORT \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --browser.gatherUsageStats false