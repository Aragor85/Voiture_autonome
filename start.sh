#!/bin/bash
set -ex

# Lancer FastAPI avec gunicorn
gunicorn app.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 127.0.0.1:8000 &

# Exporter l'URL
export API_URL="http://127.0.0.1:8000"

# Lancer Streamlit sur le port 80
streamlit run streamlit_app/app.py --server.address 0.0.0.0 --server.port 80
