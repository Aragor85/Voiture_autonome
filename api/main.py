from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.model_utils import load_model_from_url
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = FastAPI()

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
    img_array = np.array(image).astype("float32") / 255.0  # Normalisation
    img_array = np.expand_dims(img_array, axis=0)  # Shape (1, 224, 224, 3)
    return img_array

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
    prediction = model.predict(input_data)  # Shape: (1, 224, 224, num_classes)
    mask = np.argmax(prediction.squeeze(), axis=-1).astype(np.uint8)  # Shape: (224, 224)

    return {"prediction": mask.tolist()}
