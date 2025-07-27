import os
import requests
from tensorflow.keras.models import load_model

MODEL_URL = (
    "https://modelevgg16unetstorage.blob.core.windows.net/"
    "modelevgg16unetstorage/unet_vgg16_best.h5"
)
# On stocke dans /app/model pour que start.sh puisse créer ce dossier
LOCAL_MODEL_PATH = "/app/model/unet_vgg16_best.h5"

def download_model():
    os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
    if not os.path.isfile(LOCAL_MODEL_PATH):
        print("📥 Téléchargement du modèle depuis le blob…")
        r = requests.get(MODEL_URL, stream=True)
        r.raise_for_status()
        with open(LOCAL_MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
        print("✅ Modèle téléchargé.")
    else:
        print("✅ Modèle déjà présent.")

def load_model_from_url():
    download_model()
    # compile=False pour ignorer losses manquantes
    return load_model(LOCAL_MODEL_PATH, compile=False)
