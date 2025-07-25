#!/bin/bash
set -e  # Stopper si erreur

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

echo "Démarrage de l'application fusionnée (FastAPI + Streamlit)..."

# Lancer FastAPI sur le port 8000 en arrière-plan
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Lancer Streamlit sur le port 80 en arrière-plan
streamlit run app/app.py --server.port 80 --server.address 0.0.0.0 &

# Attendre que les processus restent actifs
wait
