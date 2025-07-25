#!/bin/bash
set -e  # Arrêter le script si une commande échoue

MODEL_PATH=/app/api/model/unet_vgg16_best.h5
MODEL_URL="https://modelevgg16unetstorage.blob.core.windows.net/modelevgg16unetstorage/unet_vgg16_best.h5"

# Vérifier si le modèle existe déjà
if [ ! -f "$MODEL_PATH" ]; then
  echo "Modèle non trouvé, téléchargement en cours..."
  curl -L -o "$MODEL_PATH" "$MODEL_URL"
  echo "Téléchargement terminé."
else
  echo "Modèle déjà présent, pas de téléchargement."
fi

# Lancer l'application FastAPI avec uvicorn
echo "Démarrage de l'application..."
uvicorn api.main:app --host 0.0.0.0 --port 8000
