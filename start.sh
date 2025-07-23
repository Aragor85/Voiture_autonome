#!/bin/bash
set -ex

# 1. Démarrage de FastAPI en arrière-plan sur 127.0.0.1:8000
uvicorn app.main:app --host 127.0.0.1 --port 8000 &

# 2. Exporter l'URL pour Streamlit
export API_URL="http://127.0.0.1:8000"

# 3. Démarrage de Streamlit au premier plan sur le port 80
streamlit run streamlit_app/app.py --server.address 0.0.0.0 --server.port 80
