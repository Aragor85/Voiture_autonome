# Dockerfile

FROM python:3.10

# Définir le répertoire de travail
WORKDIR /app

# Copier les dossiers nécessaires
COPY api/ /app/api/
COPY api/model/ /app/model/      
COPY app/ /app/app/
COPY requirements.txt .
COPY start.sh /app/start.sh

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Rendre le script exécutable
RUN chmod +x /app/start.sh

# Commande par défaut
CMD ["./start.sh"]
