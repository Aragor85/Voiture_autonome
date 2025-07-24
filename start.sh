#!/bin/bash
set -eux  # -e: exit on error, -u: unset var → err, -x: echo cmds

# 1. FastAPI sur toutes les interfaces, port 8000
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# 2. API_URL pour Streamlit
export API_URL="http://127.0.0.1:8000"

# 3. Streamlit en PID 1 sur le port 80
exec streamlit run app/streamlit_app.py \
     --server.address 0.0.0.0 \
     --server.port 80
