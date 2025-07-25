from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.model_utils import load_model_from_url, preprocess_image, predict
from PIL import Image
import io
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle sans recompilation
model = load_model_from_url()

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
    prediction = predict(model, input_data)  # shape: (1, H, W, num_classes)
    mask = np.argmax(prediction.squeeze(), axis=-1).astype(np.uint8)  # shape: (H, W)

    # On renvoie le masque brut (pas superposé ici car API ne fait que traiter)
    mask_list = mask.tolist()
    return {"mask": mask_list}
