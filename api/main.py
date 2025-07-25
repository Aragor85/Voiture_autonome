from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model_utils import load_model, preprocess_image, predict
from PIL import Image
import io
import numpy as np

app = FastAPI()

# CORS (si besoin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # changer selon besoin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle une fois au démarrage
MODEL_PATH = "model/unet_vgg16_best.h5"  # Chemin relatif depuis api/
model = load_model(MODEL_PATH)

@app.get("/")
async def root():
    return {"message": "API segmentation urbaine active"}

@app.post("/predict/")
async def predict_segmentation(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")

    # Lire l'image envoyée
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Erreur lors du traitement de l'image")

    # Pré-traiter l'image
    input_data = preprocess_image(image)

    # Faire la prédiction
    prediction = predict(model, input_data)

    # Convertir la prédiction en format souhaité (ex: masque binaire, JSON, base64, ...)
    # Ici on suppose que prediction est un tableau numpy, on le convertit en liste pour JSON
    prediction_list = prediction.squeeze().tolist()

    return {"prediction": prediction_list}
