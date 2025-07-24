import streamlit as st
from PIL import Image
import numpy as np
import requests
import io

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

uploaded_file = st.file_uploader("📤 Téléversez une image (jpg/png)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Appel API
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://127.0.0.1:8000/predict/", files=files)

    if response.status_code == 200:
        pred_array = np.array(response.json()["prediction"], dtype=np.uint8)

        color_mask = np.zeros((224, 224, 3), dtype=np.uint8)
        for class_id, color in enumerate(cityscapes_palette):
            color_mask[pred_array == class_id] = color

        mask_img = Image.fromarray(color_mask)
        image_resized = image.resize((224, 224))
        blended = overlay_mask(image_resized, mask_img, alpha=0.5)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image_resized, caption="🖼️ Image originale (224x224)", use_container_width=True)
        with col2:
            st.image(blended, caption="🎨 Masque superposé", use_container_width=True)

        # Boutons de téléchargement
        buf_mask = io.BytesIO()
        mask_img.save(buf_mask, format="PNG")
        st.download_button("📥 Télécharger le masque", data=buf_mask.getvalue(),
                           file_name="mask.png", mime="image/png")

        buf_blended = io.BytesIO()
        blended.save(buf_blended, format="PNG")
        st.download_button("📥 Télécharger l'image superposée", data=buf_blended.getvalue(),
                           file_name="blended.png", mime="image/png")

        # Légende
        st.markdown("### 🧭 Légende des classes")
        legend_html = ""
        for i, label in enumerate(cityscapes_labels):
            r, g, b = cityscapes_palette[i]
            hex_color = f'#{r:02x}{g:02x}{b:02x}'
            legend_html += f"<div style='display:inline-block; margin:4px;'><div style='width:20px;height:20px;background-color:{hex_color};display:inline-block;vertical-align:middle;margin-right:8px;'></div>{label}</div><br>"
        st.markdown(legend_html, unsafe_allow_html=True)

    else:
        st.error("❌ Erreur lors de la prédiction via l’API.")
