#!/bin/bash

echo "🚀 Starting BikeRisk Application..."

# Start FastAPI backend in background
echo "📡 Starting FastAPI backend on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Store the PID of FastAPI
FASTAPI_PID=$!
echo "FastAPI PID: $FASTAPI_PID"

# Wait for FastAPI to be ready
echo "⏳ Waiting for FastAPI to start..."
sleep 5

# Check if FastAPI is running
if ps -p $FASTAPI_PID > /dev/null; then
   echo "✅ FastAPI is running"
else
   echo "❌ FastAPI failed to start"
   exit 1
fi

# Start Streamlit frontend
echo "🎨 Starting Streamlit frontend on port ${PORT:-8501}..."
streamlit run app.py \
  --server.port ${PORT:-8501} \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
