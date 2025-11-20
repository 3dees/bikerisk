#!/bin/bash

set -e

echo "Starting BikeRisk Application..."

# Start FastAPI backend on port 8000
echo "Starting FastAPI backend on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info &
sleep 5

# Start Streamlit frontend on Railway's assigned PORT
echo "Starting Streamlit on port $PORT..."
streamlit run app.py \
  --server.port $PORT \
  --server.address 0.0.0.0 \
  --server.headless true &

# Wait for services to be ready
sleep 10

echo "Services started!"
echo "FastAPI: http://localhost:8000"
echo "Streamlit: http://localhost:$PORT"
echo "Keeping container alive..."

# Keep container running
tail -f /dev/null
