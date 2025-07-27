# Base légère avec Python 3.10
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer curl (pour le téléchargement du modèle)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet
COPY . .

# Rendre le script de démarrage exécutable
RUN chmod +x start.sh

# Exposer le port Streamlit (port public)
EXPOSE 8080

# Lancer le script au démarrage
CMD ["./start.sh"]
