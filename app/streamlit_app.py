import os, io, time, base64
import streamlit as st
import requests
import numpy as np
from PIL import Image

PALETTE = [
    (128, 64, 128), (220, 20, 60), (0, 0, 142), (70, 70, 70),
    (153, 153, 153), (107, 142, 35), (70, 130, 180), (0, 0, 0),
]
LABELS = ["Route/Trottoir","Humain","Véhicule","Construction",
          "Objet","Nature","Ciel","Void"]

st.set_page_config(layout="wide")
st.title("Segmentation Urbaine")
API = os.getenv("API_URL","http://localhost:8000")

f = st.file_uploader("Image", type=["jpg","png","jpeg"])
if f:
    img = Image.open(f).convert("RGB")
    st.image(img, caption="Originale")
    # appel API avec retry
    for _ in range(30):
        try:
            r = requests.post(f"{API}/predict/",
                              files={"file": (f.name, f, f.type)}, timeout=5)
            r.raise_for_status(); break
        except: time.sleep(1)
    data = r.json()
    b64 = data["mask_base64"]
    mask = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
    # recolor
    arr = np.array(mask)
    col = np.zeros((*arr.shape,3), np.uint8)
    for i,c in enumerate(PALETTE):
        col[arr==i] = c
    col_img = Image.fromarray(col)
    blend = Image.blend(img.resize(col_img.size).convert("RGBA"),
                        col_img.convert("RGBA"), alpha=0.5)
    st.image(blend, caption="Masque superposé")
    # downloads
    buf1 = io.BytesIO(); col_img.save(buf1,"PNG")
    st.download_button("Télécharger masque", buf1.getvalue(),"mask.png","image/png")
