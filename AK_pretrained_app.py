import streamlit as st
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="centered")

st.title("🌿 Plant Disease Classifier")
status = st.status("Starting…", expanded=True)
status.write("Streamlit rendered ✅")

# ---- heavy imports AFTER first render ----
status.write("Importing libraries…")
import json
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
status.write("Imports loaded ✅")

# ---- robust paths (independent of where you run from) ----
IMG_SIZE = (224, 224)
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "v3_best.keras"
CLASS_NAMES_PATH = APP_DIR / "data" / "splits" / "class_names.json"

status.write(f"Model exists: {MODEL_PATH.exists()}  ({MODEL_PATH})")
status.write(f"Class names exist: {CLASS_NAMES_PATH.exists()}  ({CLASS_NAMES_PATH})")

# ---- load class names ----
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)
status.write(f"class_names loaded ✅ ({len(class_names)} classes)")

# ---- load model (this is the slow part) ----
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, safe_mode=False)

status.write("Loading model… (first time can take a bit)")
model = load_model()
status.write("Model loaded ✅")
status.update(label="Ready", state="complete", expanded=False)

from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_HAS_INTERNAL_PREPROCESS = any(layer.name == "preprocess_input" for layer in model.layers)

def preprocess_image(img):
    img = img.convert("RGB").resize((224, 224))
    x = np.array(img).astype("float32")  # [0..255]
    if not MODEL_HAS_INTERNAL_PREPROCESS:
        x = preprocess_input(x)
    return np.expand_dims(x, axis=0)

# ---- main app ----
uploaded_file = st.file_uploader("Upload a leaf image (jpg / png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    x = preprocess_image(image)

    with st.spinner("Predicting…"):
        preds = model.predict(x, verbose=0)

    # preds is usually shape (1, 10)
    probs = preds[0] if getattr(preds, "ndim", 0) == 2 else preds
    probs = np.array(probs, dtype=float)

    pred_idx = int(np.argmax(probs))
    pred_conf = float(np.max(probs))
    pred_label = class_names[pred_idx]

    st.subheader("Prediction")
    st.write(f"**{pred_label}**")
    st.write(f"Confidence: **{pred_conf:.3f}**")

    top_indices = np.argsort(probs)[::-1][:5]
    st.subheader("Top probabilities")
    st.table({
        "Class": [class_names[i] for i in top_indices],
        "Probability": [f"{float(probs[i]):.4f}" for i in top_indices],
    })
else:
    st.info("Upload an image to start.")
