import os
import io
import time
import base64
import threading
import numpy as np
import requests
from PIL import Image
import streamlit as st

# === FastAPI backend intégré ===
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from tensorflow.keras.models import load_model

# === Constants ===
MODEL_URL = "https://modelevgg16unetstorage.blob.core.windows.net/modelevgg16unetstorage/unet_vgg16_best.h5"
MODEL_PATH = "/app/api/model/unet_vgg16_best.h5"

# === Cityscapes Palette ===
cityscapes_palette = [
    (128, 64, 128),   # 0 - flat
    (220, 20, 60),    # 1 - human
    (0, 0, 142),      # 2 - vehicle
    (70, 70, 70),     # 3 - construction
    (153, 153, 153),  # 4 - object
    (107, 142, 35),   # 5 - nature
    (70, 130, 180),   # 6 - sky
    (0, 0, 0),        # 7 - void
]
cityscapes_labels = [
    "Flat (route/trottoir)", "Humain", "Véhicule", "Construction",
    "Objet", "Nature", "Ciel", "Void"
]

# === Téléchargement modèle si absent ===
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
if not os.path.exists(MODEL_PATH):
    print("Téléchargement du modèle...")
    r = requests.get(MODEL_URL)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("Modèle téléchargé.")
else:
    print("Modèle déjà présent.")

# === Chargement modèle sans compilation ===
model = load_model(MODEL_PATH, compile=False)

# === Backend FastAPI ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    img_array = np.array(image).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

def mask_to_base64(mask: np.ndarray) -> str:
    mask_img = Image.fromarray(mask.astype(np.uint8))
    buffered = io.BytesIO()
    mask_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.post("/predict/")
async def predict_segmentation(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_data = preprocess_image(image)
    prediction = model.predict(input_data)
    mask = np.argmax(prediction.squeeze(), axis=-1).astype(np.uint8)
    return {
        "mask_base64": mask_to_base64(mask),
        "width": mask.shape[1],
        "height": mask.shape[0]
    }

# === Lancer FastAPI en thread ===
def start_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)
threading.Thread(target=start_fastapi, daemon=True).start()
time.sleep(3)  # Temps pour que l’API démarre

# === FRONT Streamlit ===
st.set_page_config(layout="wide")
st.title("🚘 FUTURE VISION TRANSPORT - Segmentation Urbaine")

uploaded_file = st.file_uploader("📤 Téléversez une image (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    uploaded_file.seek(0)
    img_bytes = uploaded_file.read()

    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        st.image(image, caption="🖼️ Image importée", use_container_width=True)
    except Exception as e:
        st.error(f"Erreur de lecture de l’image : {e}")
        st.stop()

    # Appel API FastAPI (locale)
    for i in range(300):
        try:
            files = {"file": (uploaded_file.name, io.BytesIO(img_bytes), uploaded_file.type)}
            response = requests.post("http://localhost:8000/predict/", files=files, timeout=10)
            response.raise_for_status()
            break
        except Exception:
            time.sleep(1)
    else:
        st.error("❌ L’API n’a pas répondu après 300 secondes.")
        st.stop()

    try:
        result = response.json()
        mask_data = base64.b64decode(result["mask_base64"])
        width, height = result.get("width", 224), result.get("height", 224)
        mask = Image.open(io.BytesIO(mask_data)).convert("L")
        mask_np = np.array(mask)
    except Exception as e:
        st.error(f"Erreur lecture réponse API : {e}")
        st.stop()

    # Masque coloré
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in enumerate(cityscapes_palette):
        color_mask[mask_np == class_id] = color
    color_mask_img = Image.fromarray(color_mask)

    resized_img = image.resize((width, height))
    blended = Image.blend(resized_img.convert("RGBA"), color_mask_img.convert("RGBA"), alpha=0.5)

    col1, col2 = st.columns(2)
    with col1:
        st.image(resized_img, caption="🖼️ Image redimensionnée", use_container_width=True)
    with col2:
        st.image(blended, caption="🎨 Masque superposé", use_container_width=True)

    # Téléchargement
    buf_mask = io.BytesIO()
    color_mask_img.save(buf_mask, format="PNG")
    st.download_button("📥 Télécharger le masque", buf_mask.getvalue(), file_name="mask.png", mime="image/png")

    buf_blend = io.BytesIO()
    blended.save(buf_blend, format="PNG")
    st.download_button("📥 Télécharger l'image superposée", buf_blend.getvalue(), file_name="blended.png", mime="image/png")

    # Légende
    st.markdown("### 🧭 Légende des classes")
    legend_html = ""
    for i, label in enumerate(cityscapes_labels):
        r, g, b = cityscapes_palette[i]
        legend_html += (
            f"<div style='display:inline-block;margin:4px;'>"
            f"<div style='width:20px;height:20px;background-color:#{r:02x}{g:02x}{b:02x};"
            f"display:inline-block;vertical-align:middle;margin-right:8px;'></div>"
            f"{label}</div><br>"
        )
    st.markdown(legend_html, unsafe_allow_html=True)
