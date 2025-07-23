FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# Rendre le script exécutable
RUN chmod +x start.sh

# Exposer le port que Streamlit utilise
EXPOSE 80

# Lancement
CMD ["bash", "start.sh"]
