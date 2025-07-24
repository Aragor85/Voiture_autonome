FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir -r requirements.txt -w /wheels

FROM python:3.10-slim
WORKDIR /app

# Installer netcat pour le wait-loop
RUN apt-get update \
 && apt-get install -y netcat \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl

COPY . .
RUN chmod +x start.sh

EXPOSE 80
ENTRYPOINT ["./start.sh"]
