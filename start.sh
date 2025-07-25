#!/bin/bash
set -e  # Arrêter si une erreur survient

MODEL_PATH=/app/api/model/unet_vgg16_best.h5
MODEL_URL="https://modelevgg16unetstorage.blob.core.windows.net/modelevgg16unetstorage/unet_vgg16_best.h5"

# Télécharger le modèle s'il n'existe pas
if [ ! -f "$MODEL_PATH" ]; then
  echo "Modèle non trouvé, téléchargement en cours..."
  curl -L -o "$MODEL_PATH" "$MODEL_URL"
  echo "Téléchargement terminé."
else
  echo "Modèle déjà présent, pas de téléchargement."
fi

# Lancer FastAPI en arrière-plan sur le port 8000
echo "Démarrage de FastAPI (backend) sur le port 8000..."
gunicorn api.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 &

# Exporter l'URL de l'API pour Streamlit
export API_URL="http://0.0.0.0:8000"

# Lancer Streamlit en premier plan sur le port 80 (frontend)
echo "Démarrage de Streamlit (frontend) sur le port 80..."
streamlit run app/streamlit_app.py --server.port 80 --server.address 0.0.0.0
