FROM python:3.10-slim

WORKDIR /app

# Installer curl et nettoyez le cache APT pour garder une image légère
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Télécharger le modèle (optionnel, si tu veux l’intégrer à la construction)
RUN mkdir -p /app/model && \
    curl -L "https://modelevgg16unetstorage.blob.core.windows.net/modelevgg16unetstorage/unet_vgg16_best.h5" -o /app/model/unet_vgg16_best.h5

# Copier requirements, start.sh, code
COPY requirements.txt .
COPY start.sh .
COPY api/ ./api/
COPY app/ ./app/

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

RUN chmod +x start.sh

EXPOSE 80

CMD ["./start.sh"]
