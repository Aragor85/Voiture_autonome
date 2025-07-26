FROM python:3.10-slim

WORKDIR /app

# Installer curl pour télécharger le modèle
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copier les dépendances et les installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet
COPY . .

# Rendre le script exécutable
RUN chmod +x start.sh

# Définir la variable d’environnement PORT (utile pour Azure App Service)
ENV PORT=8080

# Exposer le port utilisé par Streamlit
EXPOSE 8080

# Lancer le script de démarrage
CMD ["./start.sh"]
