from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.model_utils import load_model_from_url
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import base64

app = FastAPI()

# Autoriser les requêtes depuis n'importe quelle origine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle une fois au démarrage
model = load_model_from_url()

# Fonction de prétraitement simple (resize + normalisation)
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Convertir le masque (np.ndarray) en image PNG encodée base64
def mask_to_base64(mask: np.ndarray) -> str:
    mask_img = Image.fromarray(mask.astype(np.uint8))  # Niveaux de gris
    buffered = io.BytesIO()
    mask_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.get("/")
async def root():
    return {"message": "API segmentation urbaine active"}

@app.post("/predict/")
async def predict_segmentation(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Erreur lors du traitement de l'image")

    input_data = preprocess_image(image)
    prediction = model.predict(input_data)
    mask = np.argmax(prediction.squeeze(), axis=-1).astype(np.uint8)

    mask_base64 = mask_to_base64(mask)

    return {
        "mask_base64": mask_base64,
        "width": mask.shape[1],
        "height": mask.shape[0]
    }
