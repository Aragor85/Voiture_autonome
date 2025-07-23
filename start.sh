#!/bin/bash

# Lancer FastAPI avec gunicorn sur le port 8000 en arrière-plan
gunicorn app.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 127.0.0.1:8000 &

# Exporter l'URL de l'API pour Streamlit
export API_URL="http://127.0.0.1:8000"

# Lancer Streamlit en premier plan sur le port 80
streamlit run streamlit_app/app.py --server.address 0.0.0.0 --server.port 80
