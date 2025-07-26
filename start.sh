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

echo "Démarrage de l'application combinée FastAPI + Streamlit"
python start_app.py
