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


# ✅ Chemin du modèle
MODEL_PATH = "model/unet_vgg16_best.h5"

# ✅ Dictionnaire de custom_objects
CUSTOM_OBJECTS = {
    "dice_metric": dice_metric,
    "dice_loss_metric": dice_loss_metric,
    "cross_entropy_metric": cross_entropy_metric,
    "total_loss": total_loss,
    "mean_iou": mean_iou
}

# ✅ Chargement du modèle avec le contexte custom
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modèle introuvable à l'emplacement : {MODEL_PATH}")

with custom_object_scope(CUSTOM_OBJECTS):
    model = load_model(MODEL_PATH, compile=False)


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
