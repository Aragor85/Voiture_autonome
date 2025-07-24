#!/bin/bash
set -eux  # -e : exit on error, -u : error on unset vars, -x : log commands

# 1. Démarrage de FastAPI avec Uvicorn sur 127.0.0.1:8000
uvicorn api.main:app --host 127.0.0.1 --port 8000 &

# 2. Exporter l'URL pour Streamlit
export API_URL="http://127.0.0.1:8000"

# 3. Lancement de Streamlit sur le port 80
streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 80
