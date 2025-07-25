import numpy as np
import tensorflow as tf
from keras.saving import load_model  # Keras 3.x
from keras.utils import custom_object_scope
from PIL import Image
import io
import os
import requests

# 📥 Fonctions custom à importer
from api.losses_and_metrics import (
    dice_metric,
    dice_loss_metric,
    cross_entropy_metric,
    total_loss,
    mean_iou
)

# ✅ URL publique Azure du modèle
MODEL_URL = "https://modelevgg16unetstorage.blob.core.windows.net/modelevgg16unetstorage/unet_vgg16_best.h5"

# ✅ Chemin local où stocker le modèle téléchargé
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "unet_vgg16_best.h5")
MODEL_PATH = os.path.abspath(MODEL_PATH)

# 📥 Fonction de téléchargement si modèle absent
def download_model_if_needed():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100_000:
        print(f"[INFO] 📦 Modèle non trouvé localement. Téléchargement depuis Azure...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("[INFO] ✅ Modèle téléchargé avec succès.")
        except Exception as e:
            print(f"[ERROR] ❌ Erreur lors du téléchargement du modèle : {e}")
            raise
    else:
        print("[INFO] ✅ Modèle déjà présent localement.")

# ⬇️ Déclenche le téléchargement si besoin
download_model_if_needed()

# 🔍 Vérification fichier
print(f"[DEBUG] MODEL_PATH = {MODEL_PATH}")
print(f"[DEBUG] Exists? {os.path.isfile(MODEL_PATH)}")
print(f"[DEBUG] Size = {os.path.getsize(MODEL_PATH)} bytes")

# 🔧 Objets custom nécessaires au chargement
CUSTOM_OBJECTS = {
    "dice_metric": dice_metric,
    "dice_loss_metric": dice_loss_metric,
    "cross_entropy_metric": cross_entropy_metric,
    "total_loss": total_loss,
    "mean_iou": mean_iou
}

# ✅ Chargement du modèle
try:
    with custom_object_scope(CUSTOM_OBJECTS):
        model = load_model(MODEL_PATH, compile=False)
    print("[INFO] ✅ Modèle chargé avec succès.")
except Exception as e:
    print(f"[ERROR] ❌ Échec du chargement du modèle : {type(e).__name__} - {e}")
    raise

# ✅ Fonction de prédiction
def load_model_and_predict(image_file):
    """
    Prend une image binaire et retourne le masque prédit.
    """
    image = Image.open(io.BytesIO(image_file)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model(image_array, training=False).numpy()
    predicted_mask = np.argmax(prediction[0], axis=-1).astype(np.uint8)
    return predicted_mask
