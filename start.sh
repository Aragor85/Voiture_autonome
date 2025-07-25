#!/bin/bash
set -e

MODEL_PATH=/app/api/model/unet_vgg16_best.h5
MODEL_URL="https://modelevgg16unetstorage.blob.core.windows.net/modelevgg16unetstorage/unet_vgg16_best.h5"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Modèle non trouvé, téléchargement en cours..."
  curl -L -o "$MODEL_PATH" "$MODEL_URL"
  echo "Téléchargement terminé."
else
  echo "Modèle déjà présent, pas de téléchargement."
fi

echo "Démarrage de FastAPI en arrière-plan..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Azure définit la variable d'environnement PORT automatiquement
PORT=${PORT:-8080}
echo "Démarrage de Streamlit sur le port $PORT..."
streamlit run app/streamlit_app.py --server.port 8080