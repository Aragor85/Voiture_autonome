# 📦 Image de base légère
FROM python:3.10-slim

# 📁 Répertoire de travail
WORKDIR /app

# 🛠️ Dépendances système utiles (pour Pillow, image, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 📜 Copie des requirements
COPY requirements.txt .

# 🐍 Installation des packages Python
RUN pip install --no-cache-dir -r requirements.txt

# 📁 Copie du reste du code
COPY . .

# 🔐 Donner les droits au script de démarrage
RUN chmod +x start.sh

# 🌐 Exposer le port 80 (utile pour Azure App Service)
EXPOSE 80

# 🚀 Commande de démarrage
CMD ["./start.sh"]
