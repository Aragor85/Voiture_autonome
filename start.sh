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

# Démarrer l'application fusionnée (FastAPI + Streamlit)
echo "Démarrage de l'application fusionnée (FastAPI + Streamlit)..."
exec python3 api/main.py
