from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image, UnidentifiedImageError
from api.model_utils import load_model_and_predict

app = FastAPI(
    title="Segmentation Urbaine API",
    description="Endpoint de prédiction de masque pour Cityscapes",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "API Segmentation Urbaine - En ligne"}

@app.post("/predict/")
async def predict_mask(file: UploadFile = File(...)):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Fichier vide")

    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except UnidentifiedImageError:
        raise HTTPException(status_code=415, detail="Le fichier envoyé n'est pas une image valide")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la lecture de l'image : {e}")

    try:
        mask = load_model_and_predict(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne lors de la prédiction : {e}")

    return {"prediction": mask.tolist()}
