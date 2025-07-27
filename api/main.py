from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model_utils import load_model_from_url
from PIL import Image
import io
import base64
import numpy as np

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

model = load_model_from_url()

def preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

def mask_to_base64(mask: np.ndarray) -> str:
    img = Image.fromarray(mask.astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.get("/")
async def ping():
    return {"status": "ok"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Le fichier doit être une image")
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except:
        raise HTTPException(400, "Impossible de lire l’image")
    x = preprocess(img)
    pred = model.predict(x)
    mask = np.argmax(pred[0], axis=-1)
    b64 = mask_to_base64(mask)
    return {"mask_base64": b64, "width": mask.shape[1], "height": mask.shape[0]}
