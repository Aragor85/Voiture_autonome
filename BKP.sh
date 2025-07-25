#!/bin/bash
set -eux  # -e: exit on error, -u: unset var→err, -x: echo cmds

# 1. Démarrer FastAPI (Uvicorn) en arrière-plan sur toutes les interfaces avec debug logging
uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level debug &

# 2. Définit l'URL que Streamlit utilisera
export API_URL="http://localhost:8000"

# 3. Attendre que l'API soit joinable (max 30s) via /dev/tcp
echo "🕐 Attente de l'API sur localhost:8000…"
for i in {1..30}; do
  if (echo > /dev/tcp/localhost/8000) >/dev/null 2>&1; then
    echo "✅ API prête après $i secondes"
    break
  fi
  sleep 1
done

# 4. Lancer Streamlit sur le port 80 (PID 1)
exec streamlit run app/streamlit_app.py \
     --server.address 0.0.0.0 \
     --server.port 80
