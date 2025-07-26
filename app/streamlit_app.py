import os
import streamlit as st
from PIL import Image
import numpy as np
import requests
import io
import time
import base64

# 🎨 palette Cityscapes (8 classes)
palette = [
    (128, 64, 128), (220, 20, 60), (0, 0, 142),
    (70, 70, 70), (153, 153, 153), (107, 142, 35),
    (70, 130, 180), (0, 0, 0)
]
labels = [
    "Route/Trottoir", "Humain", "Véhicule", "Construction",
    "Objet", "Nature", "Ciel", "Void"
]

def overlay_mask(img: Image.Image, mask: Image.Image, alpha=0.5) -> Image.Image:
    img = img.convert("RGBA")
    mask = mask.convert("RGBA").resize(img.size)
    return Image.blend(img, mask, alpha)

st.set_page_config(layout="wide")
st.title("🚘 FUTURE VISION TRANSPORT — Segmentation Urbaine")

API_URL = os.getenv("API_URL", "http://localhost:8000")

uploaded = st.file_uploader("Téléversez une image JPG/PNG", type=["jpg","jpeg","png"])
if uploaded:
    data = uploaded.read()
    try:
        original = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        st.error("Erreur lecture image : " + str(e))
        st.stop()

    st.image(original, caption="Image originale", use_container_width=True)

    # appel à l'API
    with st.spinner("Envoi à l'API et génération du masque…"):
        try:
            res = requests.post(
                f"{API_URL}/predict/",
                files={"file": (uploaded.name, data, uploaded.type)},
                timeout=100
            )
            res.raise_for_status()
        except Exception as e:
            st.error("L'API n'a pas répondu : " + str(e))
            st.stop()

    payload = res.json()
    # décodage base64 en PIL.Image
    mask_bytes = base64.b64decode(payload["mask_base64"])
    mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
    mask_arr = np.array(mask_img)

    # colorisation
    color_mask = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 3), dtype=np.uint8)
    for cls, col in enumerate(palette):
        color_mask[mask_arr == cls] = col
    color_pil = Image.fromarray(color_mask)

    # redimension pour affichage côte-à-côte
    disp_orig = original.resize((payload["width"], payload["height"]))
    blended = overlay_mask(disp_orig, color_pil, alpha=0.5)

    c1, c2 = st.columns(2)
    c1.image(disp_orig, caption="Original (redim)", use_container_width=True)
    c2.image(blended, caption="Masque superposé", use_container_width=True)

    # téléchargements
    buf1 = io.BytesIO(); color_pil.save(buf1, "PNG")
    st.download_button("📥 Télécharger le masque coloré", data=buf1.getvalue(),
                       file_name="mask.png", mime="image/png")

    buf2 = io.BytesIO(); blended.save(buf2, "PNG")
    st.download_button("📥 Télécharger l'image fusionnée", data=buf2.getvalue(),
                       file_name="blended.png", mime="image/png")

    # légende
    st.markdown("### Légende des classes")
    for cls, label in enumerate(labels):
        r,g,b = palette[cls]
        st.markdown(f"<span style='display:inline-block;margin:4px;'>"
                    f"<div style='width:20px;height:20px;"
                    f"background-color:rgb({r},{g},{b});"
                    f"display:inline-block;vertical-align:middle;"
                    f"margin-right:8px;'></div>{label}</span>",
                    unsafe_allow_html=True)
