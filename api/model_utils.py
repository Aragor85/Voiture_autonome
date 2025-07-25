import os
import logging
import numpy as np
from PIL import Image
import io

# Chemin local vers le modèle
MODEL_PATH = "/app/api/model/unet_vgg16_best.h5"

# URL publique du modèle stocké dans Azure Blob Storage
MODEL_URL = "https://modelevgg16unetstorage.blob.core.windows.net/modelevgg16unetstorage/unet_vgg16_best.h5"

# Télécharger le modèle si nécessaire
def download_model_if_not_exists():
    if not os.path.isfile(MODEL_PATH):
        logging.info("📦 Modèle non trouvé localement. Téléchargement depuis Azure...")
        import requests
        r = requests.get(MODEL_URL)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            f.write(r.content)
        logging.info("✅ Modèle téléchargé avec succès.")
    else:
        logging.info("📦 Modèle déjà présent localement.")

    logging.debug(f"MODEL_PATH = {MODEL_PATH}")
    logging.debug(f"Exists? {os.path.isfile(MODEL_PATH)}")
    logging.debug(f"Size = {os.path.getsize(MODEL_PATH)} bytes")

# Charger le modèle une seule fois
def load_model():
    download_model_if_not_exists()
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    logging.info("✅ Modèle chargé avec succès.")
    return model

# Charger le modèle une fois au lancement de l'API
model = load_model()

# Fonction appelée par FastAPI pour faire une prédiction
def load_model_and_predict(image_bytes):
    # Ouvrir l'image et redimensionner
    img = Image.open(io.BytesIO(image_bytes)).resize((256, 256))
    img_array = np.array(img) / 255.0

    # Convertir en RGB si nécessaire
    if img_array.ndim == 2:  # grayscale
        img_array = np.stack([img_array]*3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[..., :3]

    # Ajouter la dimension batch
    img_array = np.expand_dims(img_array, axis=0)

    # Prédiction
    prediction = model.predict(img_array)

    # Masque final avec argmax
    mask = np.argmax(prediction[0], axis=-1).astype(np.uint8)

    return mask
