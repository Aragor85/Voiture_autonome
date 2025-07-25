FROM python:3.10

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier de dépendances et le script de démarrage
COPY requirements.txt .
COPY start.sh .

# Installer les dépendances Python sans cache
RUN pip install --no-cache-dir -r requirements.txt

# Copier les fichiers de l'API, du modèle et de Streamlit
COPY api/ ./api/
COPY app/ ./app/

# Rendre le script de démarrage exécutable
RUN chmod +x start.sh

# Définir la variable d’environnement pour le chemin du modèle
ENV MODEL_PATH=/app/api/model/unet_vgg16_best.h5

# Définir le point d’entrée
CMD ["./start.sh"]
