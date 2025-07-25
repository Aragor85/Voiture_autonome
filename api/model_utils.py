import numpy as np
import tensorflow as tf
from keras.saving import load_model  # Keras 3.x
from keras.utils import custom_object_scope
from PIL import Image
import io
import os

# 📥 Fonctions custom à importer
from api.losses_and_metrics import (
    dice_metric,
    dice_loss_metric,
    cross_entropy_metric,
    total_loss,
    mean_iou
)

# ✅ Chemin du modèle: priorise la variable d'environnement MODEL_PATH
env_model_path = os.getenv("MODEL_PATH")
if env_model_path:
    MODEL_PATH = env_model_path
else:
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "unet_vgg16_best.h5")
MODEL_PATH = os.path.abspath(MODEL_PATH)

# 🔍 Debug: afficher et vérifier la présence du fichier
print(f"[DEBUG] MODEL_PATH = {MODEL_PATH}")
print(f"[DEBUG] Exists? {os.path.isfile(MODEL_PATH)}")

# ✅ Dictionnaire de custom_objects
CUSTOM_OBJECTS = {
    "dice_metric": dice_metric,
    "dice_loss_metric": dice_loss_metric,
    "cross_entropy_metric": cross_entropy_metric,
    "total_loss": total_loss,
    "mean_iou": mean_iou
}

# ✅ Chargement du modèle avec détection d’erreurs
try:
    with custom_object_scope(CUSTOM_OBJECTS):
        model = load_model(MODEL_PATH, compile=False)
    print("[DEBUG] ✅ Modèle chargé avec succès")
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
