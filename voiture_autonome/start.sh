#!/bin/bash

# Lancer FastAPI en arrière-plan
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Lancer Streamlit (disponible sur port 8501)
streamlit run streamlit_app/app.py --server.address 0.0.0.0 --server.port 8501
