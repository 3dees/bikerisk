#!/bin/bash

echo "🚀 Starting BikeRisk Application..."

# Start FastAPI in background on port 8000
echo "📡 Starting FastAPI backend on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info &

# Wait for FastAPI
sleep 3

# Start Streamlit - PORT comes from Railway environment
echo "🎨 Starting Streamlit on port $PORT..."
exec streamlit run app.py \
  --server.port $PORT \
  --server.address 0.0.0.0 \
  --server.headless true