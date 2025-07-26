import os
import threading
import time
import base64
import io
import requests
import numpy as np
import streamlit as st
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.wsgi import WSGIMiddleware
import uvicorn
from tensorflow.keras.models import load_model
import tensorflow as tf

# =====================
# 🔧 CONFIGURATION
# =====================
MODEL_URL = "https://modelevgg16unetstorage.blob.core.windows.net/modelevgg16unetstorage/unet_vgg16_best.h5"
MODEL_PATH = "/app/model/unet_vgg16_best.h5"
API_URL = "http://localhost:8000"

cityscapes_palette = [
    (128, 64, 128), (220, 20, 60), (0, 0, 142), (70, 70, 70),
    (153, 153, 153), (107, 142, 35), (70, 130, 180), (0, 0, 0),
]
cityscapes_labels = [
    "Flat", "Humain", "Véhicule", "Construction",
    "Objet", "Nature", "Ciel", "Void"
]

# =====================
# 🚀 FASTAPI BACKEND
# =====================
app_fastapi = FastAPI()

app_fastapi.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

download_model()
model = load_model(MODEL_PATH, compile=False)

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    img_array = np.array(image).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_mask(image: Image.Image):
    input_tensor = preprocess_image(image)
    prediction = model.predict(input_tensor)
    mask = np.argmax(prediction.squeeze(), axis=-1).astype(np.uint8)
    return mask

@app_fastapi.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    mask = predict_mask(image)
    mask_img = Image.fromarray(mask.astype(np.uint8))
    buffered = io.BytesIO()
    mask_img.save(buffered, format="PNG")
    return JSONResponse({
        "mask_base64": base64.b64encode(buffered.getvalue()).decode("utf-8"),
        "width": mask.shape[1],
        "height": mask.shape[0]
    })

def start_fastapi():
    uvicorn.run(app_fastapi, host="0.0.0.0", port=8000)

threading.Thread(target=start_fastapi, daemon=True).start()

# =====================
# 🎨 STREAMLIT FRONT
# =====================
def overlay_mask(image_pil, mask_pil, alpha=0.5):
    image = image_pil.convert("RGBA")
    mask = mask_pil.convert("RGBA").resize(image.size)
    return Image.blend(image, mask, alpha=alpha)

st.set_page_config(layout="wide")
st.title("🚘 FUTURE VISION TRANSPORT - Segmentation Urbaine")

uploaded_file = st.file_uploader("📤 Téléversez une image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image importée", use_container_width=True)

    with st.spinner("⏳ Prédiction en cours..."):
        for _ in range(300):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(f"{API_URL}/predict/", files=files, timeout=10)
                response.raise_for_status()
                break
            except:
                time.sleep(1)
        else:
            st.error("❌ L'API n'a pas répondu.")
            st.stop()

        data = response.json()
        mask_data = base64.b64decode(data["mask_base64"])
        mask = Image.open(io.BytesIO(mask_data))

        mask_color = np.zeros((224, 224, 3), dtype=np.uint8)
        mask_array = np.array(mask)
        for i, color in enumerate(cityscapes_palette):
            mask_color[mask_array == i] = color
        mask_img = Image.fromarray(mask_color)

        image_resized = image.resize((224, 224))
        blended = overlay_mask(image_resized, mask_img)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image_resized, caption="Image originale", use_container_width=True)
        with col2:
            st.image(blended, caption="Masque superposé", use_container_width=True)

        # Téléchargements
        st.download_button("📥 Masque PNG", io.BytesIO(mask_img.tobytes()), "mask.png", "image/png")
        st.download_button("📥 Image superposée", io.BytesIO(blended.tobytes()), "blended.png", "image/png")

        # Légende
        st.markdown("### 🧭 Légende des classes")
        legend = ""
        for i, label in enumerate(cityscapes_labels):
            r, g, b = cityscapes_palette[i]
            legend += f"<div style='display:inline-block;margin:4px;'><div style='width:20px;height:20px;background-color:#{r:02x}{g:02x}{b:02x};display:inline-block;margin-right:8px;'></div>{label}</div><br>"
        st.markdown(legend, unsafe_allow_html=True)
