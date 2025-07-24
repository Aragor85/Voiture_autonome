#!/bin/bash
set -eux

# 1. FastAPI en fond
uvicorn api.main:app --host 127.0.0.1 --port 8000 &

export API_URL="http://127.0.0.1:8000"

# 2. Streamlit en PID 1
exec streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 80
