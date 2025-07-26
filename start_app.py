import os
import threading
import uvicorn
import streamlit.web.cli as stcli

def run_fastapi():
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000)

def run_streamlit():
    port = int(os.environ.get("PORT", 8080))
    import sys
    sys.argv = [
        "streamlit", "run", "app/streamlit_app.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0"
    ]
    stcli.main()

if __name__ == "__main__":
    # Démarrer FastAPI dans un thread en arrière-plan
    t = threading.Thread(target=run_fastapi, daemon=True)
    t.start()

    # Lancer Streamlit sur le port Azure (PORT)
    run_streamlit()
