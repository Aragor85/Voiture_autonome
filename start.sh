#!/bin/bash
set -e

# Chemin où on stocke le modèle
MODEL_PATH=/app/api/model/unet_vgg16_best.h5
MODEL_URL="https://modelevgg16unetstorage.blob.core.windows.net/.../unet_vgg16_best.h5"

mkdir -p "$(dirname "$MODEL_PATH")"

if [ ! -f "$MODEL_PATH" ]; then
  echo "📥 Téléchargement du modèle..."
  curl -L -o "$MODEL_PATH" "$MODEL_URL"
  echo "✅ Modèle téléchargé."
else
  echo "✅ Modèle déjà présent."
fi

echo "🚀 Lancement de l'API FastAPI..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

PORT=${PORT:-8080}
echo "🚀 Lancement de Streamlit sur le port $PORT..."
streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
