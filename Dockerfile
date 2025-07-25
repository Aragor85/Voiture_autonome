# Image Python légère
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier uniquement les fichiers nécessaires
COPY requirements.txt .
COPY start.sh .
COPY api/ ./api/
COPY app/ ./app/

# Installer les dépendances
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Donner les droits d'exécution au script de démarrage
RUN chmod +x start.sh

# Streamlit doit écouter sur le port 80
ENV PORT 80

# Exposer le port 80 (Streamlit)
EXPOSE 80

# Lancer le script de démarrage
CMD ["./start.sh"]
