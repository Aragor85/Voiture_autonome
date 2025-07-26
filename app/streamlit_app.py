import streamlit as st
from PIL import Image
import numpy as np
import requests
import base64
import io
import os
import time

API_URL = os.getenv("API_URL", "http://localhost:8000")

palette = [
    (128, 64, 128), (220, 20, 60), (0, 0, 142), (70, 70, 70),
    (153, 153, 153), (107, 142, 35), (70, 130, 180), (0, 0, 0)
]
labels = [
    "Flat", "Humain", "Véhicule", "Construction", 
    "Objet", "Nature", "Ciel", "Void"
]

st.set_page_config(layout="wide")
st.title("🚘 Segmentation Urbaine – Voiture Autonome")

uploaded_file = st.file_uploader("📤 Chargez une image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    uploaded_file.seek(0)
    img_bytes = uploaded_file.read()

    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        st.image(image, caption="🖼️ Image d'entrée", use_container_width=True)
    except:
        st.error("❌ Erreur de lecture de l’image")
        st.stop()

    # Appel API
    for _ in range(60):
        try:
            files = {"file": (uploaded_file.name, io.BytesIO(img_bytes), "image/jpeg")}
            response = requests.post(f"{API_URL}/predict/", files=files, timeout=10)
            response.raise_for_status()
            break
        except Exception:
            time.sleep(1)
    else:
        st.error("❌ API non disponible après 60 sec")
        st.stop()

    data = response.json()
    mask_b64 = data["mask_base64"]
    width, height = data["width"], data["height"]
    mask_bytes = base64.b64decode(mask_b64)
    mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
    mask_arr = np.array(mask_img)

    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for i, color in enumerate(palette):
        color_mask[mask_arr == i] = color

    blended = Image.blend(image.resize((width, height)).convert("RGBA"),
                          Image.fromarray(color_mask).convert("RGBA"),
                          alpha=0.5)

    col1, col2 = st.columns(2)
    col1.image(image.resize((width, height)), caption="Originale")
    col2.image(blended, caption="Masque superposé")

    st.download_button("📥 Télécharger le masque", data=mask_bytes,
                       file_name="mask.png", mime="image/png")

    st.markdown("### 🧭 Légende des classes")
    for label, color in zip(labels, palette):
        r, g, b = color
        st.markdown(f"<div style='display:flex; align-items:center;'>"
                    f"<div style='width:20px; height:20px; background-color:rgb({r},{g},{b}); margin-right:10px;'></div>"
                    f"{label}</div>", unsafe_allow_html=True)
