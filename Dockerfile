@'
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y `
    build-essential `
    libglib2.0-0 `
    libsm6 `
    libxext6 `
    libxrender-dev `
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x start.sh

EXPOSE 80

CMD ["./start.sh"]
'@ | Out-File -FilePath Dockerfile -Encoding utf8
