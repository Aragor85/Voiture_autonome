import threading
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.main import app as fastapi_app
import streamlit.web.bootstrap
import os

def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

def run_streamlit():
    # Utilise le même script Streamlit que d’habitude
    streamlit_args = ["streamlit_app.py", "--server.port=80", "--server.address=0.0.0.0"]
    streamlit.web.bootstrap.run(streamlit_args)

if __name__ == "__main__":
    threading.Thread(target=run_fastapi, daemon=True).start()
    run_streamlit()
