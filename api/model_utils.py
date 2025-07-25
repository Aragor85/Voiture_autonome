import os
import logging

MODEL_PATH = "/app/api/model/unet_vgg16_best.h5"
MODEL_URL = "https://modelevgg16unetstorage.blob.core.windows.net/modelevgg16unetstorage/unet_vgg16_best.h5"

def download_model_if_not_exists():
    if not os.path.isfile(MODEL_PATH):
        logging.info("📦 Modèle non trouvé localement. Téléchargement depuis Azure...")
        import requests
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(r.content)
        logging.info("✅ Modèle téléchargé avec succès.")
    else:
        logging.info("📦 Modèle déjà présent localement.")
    logging.debug(f"MODEL_PATH = {MODEL_PATH}")
    logging.debug(f"Exists? {os.path.isfile(MODEL_PATH)}")
    logging.debug(f"Size = {os.path.getsize(MODEL_PATH)} bytes")

def load_model():
    download_model_if_not_exists()
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    logging.info("✅ Modèle chargé avec succès.")
    return model
