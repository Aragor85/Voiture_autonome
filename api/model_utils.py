import os
import requests
from tensorflow.keras.models import load_model

# 📦 URL du modèle stocké dans Azure Blob (à adapter selon ton besoin)
MODEL_URL = "https://modelevgg16unetstorage.blob.core.windows.net/modelevgg16unetstorage/unet_vgg16_best_inference.h5"

# 📍 Chemin local où sera stocké le modèle
LOCAL_MODEL_PATH = "/app/api/model/unet_vgg16_best.h5"

def download_model():
    """
    Télécharge le modèle depuis Azure Blob Storage s'il n'est pas déjà présent localement.
    """
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("📦 Téléchargement du modèle en cours...")
        os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)

        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()

        with open(LOCAL_MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Modèle téléchargé.")
    else:
        print("✅ Modèle déjà présent localement.")

def load_model_from_url():
    """
    Télécharge (si besoin) et charge le modèle Keras en ignorant la compilation.
    """
    download_model()
    print("📥 Chargement du modèle pour l'inférence...")
    return load_model(LOCAL_MODEL_PATH, compile=False)
