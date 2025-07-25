from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.model_utils import load_model_from_url, preprocess_image, predict
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle depuis l'URL au démarrage
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
    prediction = predict(model, input_data)
    prediction_list = prediction.squeeze().tolist()
    return {"prediction": prediction_list}
