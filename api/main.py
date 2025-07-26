from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import os
import base64
import requests

MODEL_URL = "https://modelevgg16unetstorage.blob.core.windows.net/modelevgg16unetstorage/unet_vgg16_best.h5"
MODEL_PATH = "/app/model/unet_vgg16_best.h5"

# FastAPI instance
app = FastAPI()

# CORS pour permettre l'accès depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Téléchargement du modèle si non présent
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Téléchargement du modèle...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("✅ Modèle téléchargé.")
    else:
        print("✅ Modèle déjà présent.")

# Chargement robuste du modèle
def load_model_safely():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable à : {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Modèle chargé avec succès.")
        return model
    except Exception as e:
        raise RuntimeError(f"❌ Erreur lors du chargement du modèle : {e}")

# Chargement au démarrage
download_model()
model = load_model_safely()

# Prétraitement image (resize + normalisation)
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    array = np.array(image).astype("float32") / 255.0
    return np.expand_dims(array, axis=0)  # (1, 224, 224, 3)

# Conversion masque en base64 PNG
def mask_to_base64(mask: np.ndarray) -> str:
    img = Image.fromarray(mask.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# Endpoints
@app.get("/")
async def root():
    return {"message": "✅ API segmentation urbaine opérationnelle"}

@app.post("/predict/")
async def predict_segmentation(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="❌ Erreur lors du traitement de l'image")

    input_data = preprocess_image(image)
    prediction = model.predict(input_data)
    mask = np.argmax(prediction.squeeze(), axis=-1).astype(np.uint8)
    mask_base64 = mask_to_base64(mask)

    return {
        "mask_base64": mask_base64,
        "width": mask.shape[1],
        "height": mask.shape[0]
    }
