FROM python:3.10

# Définir le répertoire de travail
WORKDIR /app

# Copier les dossiers de l'API, du modèle et de l'interface Streamlit
COPY api/ /app/api/
COPY api/model/ /app/api/model/
COPY app/ /app/app/

# Copier le fichier de dépendances et le script de démarrage
COPY requirements.txt ./
COPY start.sh /app/start.sh

# Spécifier le chemin du modèle dans la variable d'environnement
ENV MODEL_PATH=/app/api/model/unet_vgg16_best.h5

# Installer les dépendances Python sans cache
RUN pip install --no-cache-dir -r requirements.txt

# Rendre le script de démarrage exécutable
RUN chmod +x /app/start.sh

# Lancer le script de démarrage par défaut
CMD ["/app/start.sh"]
