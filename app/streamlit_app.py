import streamlit as st
from PIL import Image
import numpy as np
import requests
import io
import os

# 📍 URL de l'API FastAPI (par défaut : local)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# ✅ Palette Cityscapes regroupée par catégories (8 classes)
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

# ✅ Utilitaire pour superposer masque et image
def overlay_mask(image_pil, mask_pil, alpha=0.5):
    image = image_pil.convert("RGBA")
    mask = mask_pil.convert("RGBA")
    if image.size != mask.size:
        mask = mask.resize(image.size)
    return Image.blend(image, mask, alpha=alpha)

# ✅ Interface utilisateur
st.set_page_config(layout="wide")
st.title("🚘 FUTURE VISION TRANSPORT - Segmentation Urbaine")

uploaded_file = st.file_uploader("📤 Téléversez une image (jpg/png)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Image originale", use_container_width=True)

    with st.spinner("⏳ Prédiction en cours..."):
        files = {"file": uploaded_file.getvalue()}
        try:
            response = requests.post(f"{API_URL}/predict/", files=files)
            response.raise_for_status()
        except Exception as e:
            st.error(f"❌ Erreur d'appel à l'API : {e}")
            st.stop()

    pred_array = np.array(response.json()["prediction"], dtype=np.uint8)

    # 🎨 Masque coloré
    color_mask = np.zeros((224, 224, 3), dtype=np.uint8)
    for class_id, color in enumerate(cityscapes_palette):
        color_mask[pred_array == class_id] = color
    mask_img = Image.fromarray(color_mask)

    # 🔄 Superposition
    blended = overlay_mask(image.resize((224, 224)), mask_img, alpha=0.5)

    st.image(mask_img, caption="🎨 Masque coloré", use_container_width=True)
    st.image(blended, caption="🔄 Superposition image + masque", use_container_width=True)

    # 📥 Téléchargement
    st.download_button("📥 Télécharger le masque",
                       data=io.BytesIO(np.array(mask_img).tobytes()),
                       file_name="mask.png", mime="image/png")
    st.download_button("📥 Télécharger l'image superposée",
                       data=io.BytesIO(np.array(blended).tobytes()),
                       file_name="blended.png", mime="image/png")

    # 🧾 Légende
    st.markdown("### 🧭 Légende des classes")
    legend_html = ""
    for i, label in enumerate(cityscapes_labels):
        r, g, b = cityscapes_palette[i]
        hex_color = f'#{r:02x}{g:02x}{b:02x}'
        legend_html += f"<div style='display:inline-block; margin:4px;'><div style='width:20px;height:20px;background-color:{hex_color};display:inline-block;vertical-align:middle;margin-right:8px;'></div>{label}</div><br>"
    st.markdown(legend_html, unsafe_allow_html=True)
