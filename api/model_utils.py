import requests
from tensorflow.keras.models import load_model
import os

MODEL_URL = "https://modelevgg16unetstorage.blob.core.windows.net/modelevgg16unetstorage/unet_vgg16_best.h5"
LOCAL_MODEL_PATH = "/app/model/unet_vgg16_best.h5"  # ou un chemin adapté dans le conteneur

def download_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("Téléchargement du modèle depuis l'URL...")
        r = requests.get(MODEL_URL)
        r.raise_for_status()
        os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
        with open(LOCAL_MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Modèle téléchargé.")
    else:
        print("Modèle déjà présent localement.")

def load_model_from_url():
    download_model()
    return load_model(LOCAL_MODEL_PATH)
