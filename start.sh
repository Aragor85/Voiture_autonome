#!/bin/bash

# Lancer FastAPI en arrière-plan
gunicorn app.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 &

# Définir l'URL de l'API accessible pour Streamlit
export API_URL="http://127.0.0.1:8000"

# Lancer Streamlit en premier plan (sur port 80)
streamlit run streamlit_app/app.py --server.address 0.0.0.0 --server.port 80
