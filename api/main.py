from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import base64
import os
from tensorflow.keras.models import load_model

app = FastAPI()

# CORS pour autoriser les appels Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "app/model/unet_vgg16_best.h5"

# Charger le modèle une fois
model = load_model(MODEL_PATH, compile=False)

def preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    array = np.array(image).astype("float32") / 255.0
    return np.expand_dims(array, axis=0)

def mask_to_base64(mask: np.ndarray) -> str:
    mask_img = Image.fromarray(mask.astype(np.uint8))
    buffer = io.BytesIO()
    mask_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.get("/")
def root():
    return {"message": "API FastAPI prête 🚀"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Format non valide")

    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Erreur d’ouverture d’image")

    input_array = preprocess(image)
    prediction = model.predict(input_array)
    mask = np.argmax(prediction.squeeze(), axis=-1)

    return {
        "mask_base64": mask_to_base64(mask),
        "width": mask.shape[1],
        "height": mask.shape[0]
    }
