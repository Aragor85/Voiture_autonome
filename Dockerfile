FROM python:3.10

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances et le script de démarrage
COPY requirements.txt .
COPY start.sh .

# Installer les dépendances Python sans cache
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY api/ ./api/
COPY app/ ./app/

# Télécharger le fichier du modèle depuis Azure Blob Storage
RUN curl -o ./api/model/unet_vgg16_best.h5 https://modelevgg16unetstorage.blob.core.windows.net/modelevgg16unetstorage/unet_vgg16_best.h5

# Rendre le script de démarrage exécutable
RUN chmod +x start.sh

# Définir la variable d’environnement pour le chemin du modèle
ENV MODEL_PATH=/app/api/model/unet_vgg16_best.h5

# Exposer le port utilisé (ex: par FastAPI ou Streamlit)
EXPOSE 8000

# Point d’entrée
CMD ["./start.sh"]
