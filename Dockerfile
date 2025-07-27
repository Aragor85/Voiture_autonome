# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /app

# 1) Installer curl et nettoyer le cache
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# 2) Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copier votre API et votre front
COPY api/ ./api/
COPY app/ ./app/

# 4) Copier le script de démarrage
COPY start.sh .
RUN chmod +x start.sh

# 5) Exposer le port Streamlit (8080)
EXPOSE 8080

# 6) Lancer start.sh qui démarre FastAPI + Streamlit
CMD ["./start.sh"]
