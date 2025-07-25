FROM python:3.10-slim

WORKDIR /app

# Installer curl pour downloader le modèle
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x start.sh

EXPOSE 8080

CMD ["./start.sh"]
