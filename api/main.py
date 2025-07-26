from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.model_utils import load_model_from_url
from PIL import Image
import io
import numpy as np
import base64

app = FastAPI(
    title="Segmentation Urbaine API",
    description="Charge un modèle depuis une URL et renvoie un masque PNG encodé en Base64"
)

# CORS ouvert pour tester (à restreindre en prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement unique du modèle au démarrage
model = load_model_from_url()

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize + normalisation pour inference."""
    image = image.resize((224, 224))
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)  # shape (1,224,224,3)

def mask_to_base64(mask: np.ndarray) -> str:
    """
    Convertit un masque 2D uint8 en image PNG base64.
    Toutes les valeurs de mask sont interprétées comme niveaux de gris.
    """
    pil = Image.fromarray(mask, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

@app.get("/")
async def root():
    return {"message": "API Segmentation Urbaine OK"}

@app.post("/predict/")
async def predict_segmentation(file: UploadFile = File(...)):
    # Vérifier le type MIME
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Le fichier doit être une image")

    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Impossible de décoder l'image")

    x = preprocess_image(img)
    preds = model.predict(x)  # shape (1,224,224,num_classes)
    mask = np.argmax(preds[0], axis=-1).astype(np.uint8)  # (224,224)

    b64 = mask_to_base64(mask)
    h, w = mask.shape

    return {
        "mask_base64": b64,
        "width": w,
        "height": h
    }
