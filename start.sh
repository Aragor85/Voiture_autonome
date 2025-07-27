#!/usr/bin/env bash
set -e

# 1) Télécharger le modèle si absent
MODEL_PATH=/app/model/unet_vgg16_best.h5
MODEL_URL="https://modelevgg16unetstorage.blob.core.windows.net/modelevgg16unetstorage/unet_vgg16_best.h5"
mkdir -p "$(dirname "$MODEL_PATH")"
if [ ! -f "$MODEL_PATH" ]; then
  echo "📥 Téléchargement du modèle…"
  curl -L -o "$MODEL_PATH" "$MODEL_URL"
  echo "✅ Modèle prêt."
fi

# 2) Lancer FastAPI
echo "🚀 Démarrage FastAPI…"
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# 3) Lancer Streamlit
PORT=${PORT:-8080}
echo "🚀 Démarrage Streamlit sur $PORT…"
streamlit run app/streamlit_app.py --server.port "$PORT" --server.address 0.0.0.0
