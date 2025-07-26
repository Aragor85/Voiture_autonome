import os
import streamlit as st
from PIL import Image
import numpy as np
import requests
import io
import time
import base64

# ✅ Colormap Cityscapes (groupée en 8 catégories)
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

def overlay_mask(image_pil, mask_pil, alpha=0.5):
    image = image_pil.convert("RGBA")
    mask = mask_pil.convert("RGBA")
    if image.size != mask.size:
        mask = mask.resize(image.size)
    return Image.blend(image, mask, alpha=alpha)

st.set_page_config(layout="wide")
st.title("🚘 FUTURE VISION TRANSPORT - Segmentation Urbaine - Voiture Autonome")

API_URL = os.getenv("API_URL", "http://localhost:8000")
uploaded_file = st.file_uploader("📤 Téléversez une image (jpg/png)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    uploaded_file.seek(0)
    img_bytes = uploaded_file.read()

    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        st.image(image, caption="🖼️ Image importée", use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lecture image : {e}")
        st.stop()

    max_wait = 300
    ext = uploaded_file.name.split('.')[-1].lower()
    mime = 'image/png' if ext == 'png' else 'image/jpeg'

    for i in range(1, max_wait + 1):
        files = {"file": (uploaded_file.name, io.BytesIO(img_bytes), mime)}
        try:
            response = requests.post(
                f"{API_URL}/predict/",
                files=files,
                timeout=5
            )
            response.raise_for_status()
            break
        except requests.exceptions.RequestException:
            time.sleep(1)
    else:
        st.error(f"❌ L’API n’a pas répondu après {max_wait} secondes.")
        st.stop()

    # ✅ Traitement de la réponse base64
    try:
        response_json = response.json()
        mask_base64 = response_json["mask_base64"]
        width = response_json.get("width", 224)
        height = response_json.get("height", 224)

        # Reconstruction du masque en niveaux de gris
        mask_bytes = base64.b64decode(mask_base64)
        mask_gray = Image.open(io.BytesIO(mask_bytes)).convert("L")
        pred_array = np.array(mask_gray)

    except Exception as e:
        st.error(f"Réponse API invalide : {e}")
        st.stop()

    # ✅ Générer le masque couleur
    color_mask = np.zeros((pred_array.shape[0], pred_array.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(cityscapes_palette):
        color_mask[pred_array == class_id] = color

    mask_img = Image.fromarray(color_mask)
    image_resized = image.resize(mask_img.size)
    blended = overlay_mask(image_resized, mask_img, alpha=0.5)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image_resized, caption="🖼️ Image originale (redimensionnée)", use_container_width=True)
    with col2:
        st.image(blended, caption="🎨 Masque superposé", use_container_width=True)

    # ✅ Boutons de téléchargement
    buf_mask = io.BytesIO()
    mask_img.save(buf_mask, format="PNG")
    st.download_button("📥 Télécharger le masque", data=buf_mask.getvalue(),
                       file_name="mask.png", mime="image/png")

    buf_blended = io.BytesIO()
    blended.save(buf_blended, format="PNG")
    st.download_button("📥 Télécharger l'image superposée", data=buf_blended.getvalue(),
                       file_name="blended.png", mime="image/png")

    # ✅ Légende
    st.markdown("### 🧭 Légende des classes")
    legend_html = ""
    for i, label in enumerate(cityscapes_labels):
        r, g, b = cityscapes_palette[i]
        hex_color = f'#{r:02x}{g:02x}{b:02x}'
        legend_html += (
            f"<div style='display:inline-block; margin:4px;'>"
            f"<div style='width:20px;height:20px;background-color:{hex_color};"
            f"display:inline-block;vertical-align:middle;margin-right:8px;'></div>"
            f"{label}</div><br>"
        )
    st.markdown(legend_html, unsafe_allow_html=True)
