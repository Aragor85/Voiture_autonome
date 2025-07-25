#!/usr/bin/env bash
set -e

# 🔧 Définition de l'URL de l'API
export API_URL=${API_URL:-http://localhost:8000}
echo "🕐 Attente de l'API sur ${API_URL}…"

# ── Lancement de l'API en arrière-plan
uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level debug &
API_PID=$!

# ── Boucle de vérification de disponibilité de l'API
max_wait=30
for i in $(seq 1 $max_wait); do
    if curl -sSf "${API_URL}/docs" > /dev/null; then
        echo "✅ API prête après ${i} secondes"
        break
    fi
    sleep 1
done

# ── Si l'API n'est pas prête, on affiche une erreur et on stoppe
if ! curl -sSf "${API_URL}/docs" > /dev/null; then
    echo "❌ L’API n’a pas répondu après ${max_wait} secondes. Abandon."
    kill $API_PID
    exit 1
fi

# ── Lancement de Streamlit au premier-plan (remplace le shell)
exec streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 80
