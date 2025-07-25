FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY start.sh .

RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY app/ ./app/

RUN mkdir -p ./api/model

RUN chmod +x start.sh

ENV MODEL_PATH=/app/api/model/unet_vgg16_best.h5

EXPOSE 8000

CMD ["./start.sh"]
