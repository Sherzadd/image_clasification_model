# app.py
# ✅ No user settings shown
# ✅ Sidebar = User Manual only
# ✅ Uses camera button ("Take a photo") + optional upload
# ✅ Auto-predict immediately after photo/upload (no Predict button)
# ✅ Loads class names from embedded ClassNamesLayer inside the model
# ✅ Robust model loading (safe_mode fallback + patched Conv2D)
# ✅ Caches model and reloads automatically when model file changes

from pathlib import Path
from typing import List, Optional
import hashlib
import io

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_REL_PATH = "models/image_classification_model_with_names.keras"
MODEL_PATH_TEXT = DEFAULT_MODEL_REL_PATH


def resolve_path(text: str) -> Path:
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()
    return p


# -----------------------------
# Embedded class names layer
# -----------------------------
@tf.keras.utils.register_keras_serializable(package="meta")
class ClassNamesLayer(tf.keras.layers.Layer):
    """
    Metadata-only layer:
    - stores class_names inside the saved .keras model
    - identity forward pass (does not change inputs)
    - robust to deserialization
    """
    def __init__(self, class_names=None, **kwargs):
        # Avoid "trainable passed twice" issues
        kwargs.pop("trainable", None)
        super().__init__(**kwargs)
        self.class_names = list(class_names) if class_names is not None else []
        self.trainable = False

    def call(self, inputs):
        return inputs

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"class_names": self.class_names})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# -----------------------------
# Robust model loading
# -----------------------------
def _load_model_call(
    path: Path, *, compile: bool, safe_mode: Optional[bool], custom_objects: Optional[dict]
):
    kwargs = {"compile": compile}
    if custom_objects is not None:
        kwargs["custom_objects"] = custom_objects

    if safe_mode is not None:
        try:
            return tf.keras.models.load_model(path, **kwargs, safe_mode=safe_mode)
        except TypeError:
            return tf.keras.models.load_model(path, **kwargs)

    return tf.keras.models.load_model(path, **kwargs)


@st.cache_resource
def load_model(model_path_text: str, model_mtime: float):
    model_path = resolve_path(model_path_text)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    @tf.keras.utils.register_keras_serializable(package="Patched")
    class PatchedConv2D(tf.keras.layers.Conv2D):
        def __init__(self, *args, batch_input_shape=None, **kwargs):
            # Ignore legacy batch_input_shape if present
            super().__init__(*args, **kwargs)

    custom = {
        "Conv2D": PatchedConv2D,
        "keras.layers.Conv2D": PatchedConv2D,
        "keras.src.layers.convolutional.conv2d.Conv2D": PatchedConv2D,
        "ClassNamesLayer": ClassNamesLayer,
        "meta.ClassNamesLayer": ClassNamesLayer,
    }

    # Try normal first, then safe_mode=False + custom
    try:
        return _load_model_call(model_path, compile=False, safe_mode=None, custom_objects=None)
    except Exception:
        return _load_model_call(model_path, compile=False, safe_mode=False, custom_objects=custom)


def get_class_names_from_model(model) -> List[str]:
    def walk(layer) -> Optional[List[str]]:
        if isinstance(layer, ClassNamesLayer):
            return list(layer.class_names)
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                got = walk(sub)
                if got:
                    return got
        return None

    return walk(model) or []


# -----------------------------
# Helpers
# -----------------------------
def model_input_size(model) -> int:
    try:
        shp = model.input_shape  # (None, H, W, C)
        if isinstance(shp, (list, tuple)) and len(shp) == 4 and shp[1] and shp[2]:
            return int(shp[1])
    except Exception:
        pass
    return 256


def preprocess_image_0_255(pil_img: Image.Image, size: int) -> np.ndarray:
    img = pil_img.convert("RGB").resize((size, size))
    arr = np.array(img).astype("float32")
    return np.expand_dims(arr, axis=0)


def to_probs(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).astype("float32")
    if x.ndim == 2 and x.shape[0] == 1:
        x = x[0]

    s = float(np.sum(x))
    if 0.98 <= s <= 1.02 and np.all(x >= 0) and np.all(x <= 1):
        return x

    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="wide")
st.title("🌿 Plant Disease Classifier")
st.caption("Take a photo (recommended) or upload a leaf image — prediction runs automatically.")

# Sidebar = User Manual only
st.sidebar.title("📘 User Manual")
st.sidebar.markdown(
    """
**How to take a good photo (important):**
- Use bright natural light (avoid very dark photos).
- Keep the leaf **in focus** (no blur).
- Capture **one leaf clearly** (fill most of the frame).
- Use a **plain background** if possible.
- Avoid strong shadows, reflections, and filters.
- Don’t crop too tightly — include the full infected area.

**How to use the app:**
1. Click **Take a photo** OR upload an image  
2. Wait ~1 second → prediction appears automatically
"""
)

# Load model silently (no “Model ready” message)
model_path = resolve_path(MODEL_PATH_TEXT)
if not model_path.exists():
    st.error("Model file is missing in the deployed app. Please add it to the repository.")
    st.stop()

mtime = model_path.stat().st_mtime

try:
    model = load_model(MODEL_PATH_TEXT, mtime)
except Exception as e:
    st.error("❌ Model found, but failed to load.")
    st.exception(e)
    st.stop()

class_names = get_class_names_from_model(model)
out_dim = None
try:
    out_dim = int(model.output_shape[-1])
except Exception:
    pass

if not class_names:
    class_names = [f"class_{i}" for i in range(out_dim or 0)]

# Main inputs
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📷 Capture or upload")
    cam = st.camera_input("Take a photo")
    up = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg", "webp"])

    file_obj = cam if cam is not None else up

with col2:
    st.subheader("✅ Results")

    if file_obj is None:
        st.info("No image yet. Use **Take a photo** or upload an image.")
        st.stop()

    img_bytes = file_obj.getvalue()
    img_hash = sha1_bytes(img_bytes)

    # Show image
    pil = Image.open(io.BytesIO(img_bytes))
    st.image(pil, caption="Your image", use_container_width=True)

    # Auto-predict only when image changes
    if st.session_state.get("last_img_hash") != img_hash:
        st.session_state["last_img_hash"] = img_hash

        size = model_input_size(model)
        batch = preprocess_image_0_255(pil, size)

        with st.spinner("Predicting..."):
            raw = model.predict(batch, verbose=0)
            probs = to_probs(raw)

        st.session_state["last_probs"] = probs

    probs = st.session_state.get("last_probs", None)
    if probs is None:
        st.warning("Prediction not available yet.")
        st.stop()

    best_idx = int(np.argmax(probs))
    best_name = class_names[best_idx] if best_idx < len(class_names) else f"class_{best_idx}"
    best_conf = float(probs[best_idx]) * 100

    st.markdown("### Prediction")
    st.markdown(f"**{best_name}**")
    st.metric("Confidence", f"{best_conf:.2f}%")

    st.markdown("### Top predictions")
    top_k = min(5, len(probs))
    top_idx = np.argsort(probs)[::-1][:top_k]
    rows = []
    for i in top_idx:
        name = class_names[i] if i < len(class_names) else f"class_{i}"
        rows.append({"Class": name, "Probability (%)": float(probs[i]) * 100})

    st.dataframe(rows, use_container_width=True, hide_index=True)
