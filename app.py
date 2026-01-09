# app.py
# ✅ User-friendly UI (no technical settings)
# ✅ Sidebar shows a User Manual instead of model settings
# ✅ Class names auto-loaded from MODEL (embedded ClassNamesLayer)
# ✅ Reloads automatically when model file is updated (mtime cache key)
# ✅ Robust loading with patched Conv2D + safe_mode=False fallback
# ✅ Clean prediction output + Top-5 table

from pathlib import Path
from typing import List, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf


# -----------------------------
# Fixed model path (hidden from users)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_REL_PATH = "models/image_classification_model_with_names.keras"


def resolve_path(text: str) -> Path:
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()
    return p


# -----------------------------
# Embedded class names layer
# (must exist in app to load model)
# -----------------------------
@tf.keras.utils.register_keras_serializable(package="meta")
class ClassNamesLayer(tf.keras.layers.Layer):
    """
    Metadata-only layer:
    - stores class_names inside the saved .keras model
    - identity forward pass (does not change inputs)
    - robust to Keras deserialization (trainable conflicts avoided)
    """
    def __init__(self, class_names=None, **kwargs):
        kwargs.pop("trainable", None)   # avoid trainable passed twice
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
    """
    model_mtime is included so Streamlit cache invalidates when you overwrite the file.
    """
    model_path = resolve_path(model_path_text)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    @tf.keras.utils.register_keras_serializable(package="Patched")
    class PatchedConv2D(tf.keras.layers.Conv2D):
        def __init__(self, *args, batch_input_shape=None, **kwargs):
            super().__init__(*args, **kwargs)

    custom = {
        "Conv2D": PatchedConv2D,
        "keras.layers.Conv2D": PatchedConv2D,
        "keras.src.layers.convolutional.conv2d.Conv2D": PatchedConv2D,
        "ClassNamesLayer": ClassNamesLayer,
        "meta.ClassNamesLayer": ClassNamesLayer,
    }

    attempts = [
        dict(safe_mode=None, custom_objects=custom),
        dict(safe_mode=False, custom_objects=custom),
        dict(safe_mode=None, custom_objects=None),
        dict(safe_mode=False, custom_objects=None),
    ]

    last_err = None
    for a in attempts:
        try:
            return _load_model_call(
                model_path,
                compile=False,
                safe_mode=a["safe_mode"],
                custom_objects=a["custom_objects"],
            )
        except Exception as e:
            last_err = e

    raise last_err


# -----------------------------
# Read class names from model
# -----------------------------
def get_class_names_from_model(model) -> List[str]:
    def walk(layer) -> Optional[List[str]]:
        if hasattr(layer, "class_names"):
            try:
                names = list(layer.class_names)
                if names:
                    return names
            except Exception:
                pass

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
    pil_img = ImageOps.exif_transpose(pil_img)  # fixes rotated phone photos
    img = pil_img.convert("RGB").resize((size, size))
    arr = np.array(img).astype("float32")      # 0..255
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


def format_percent(p: float) -> str:
    return f"{p * 100:.2f}%"


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="wide")

st.title("🌿 Plant Disease Classifier")
st.caption("Upload a clear leaf photo and get the predicted plant disease class from your trained model.")

# ---- Sidebar: User Manual (instead of settings)
st.sidebar.title("📘 User Manual")

st.sidebar.markdown(
    """
**How to take a good photo (important):**
- Use **bright natural light** (avoid very dark photos).
- Keep the leaf **in focus** (no blur).
- Capture **one leaf clearly** (fill most of the frame).
- Use a **plain background** if possible.
- Avoid **strong shadows**, reflections, and filters.
- Don’t crop too tightly — include the full infected area.

**How to use the app:**
1) Upload a leaf image  
2) Click **Predict**  
3) Read the prediction + Top-5 results

**Supported files:** PNG, JPG/JPEG, WEBP
"""
)

with st.sidebar.expander("Troubleshooting"):
    st.markdown(
        """
- If you see **class_0 / class_1** instead of real names, you are probably using a model
  that does not include embedded class names. Use the `*_with_names.keras` file.
- If prediction confidence looks wrong, verify your training preprocessing:
  - If your model contains `Rescaling(1./255)` inside, this app is correct (sends 0..255).
  - If your model expects already-scaled 0..1 input, you must add scaling here.
"""
    )

st.sidebar.caption("Tip: Use photos similar to your training data for best accuracy.")

# ---- Hidden model path (no user input)
MODEL_PATH_TEXT = DEFAULT_MODEL_REL_PATH
model_path = resolve_path(MODEL_PATH_TEXT)

if not model_path.exists():
    st.error("❌ Model file not found.")
    st.write("Expected path:", str(model_path))
    st.info("Make sure your model is saved inside your project folder under /models.")
    st.stop()

mtime = model_path.stat().st_mtime

# Load model
try:
    with st.spinner("Loading model..."):
        model = load_model(MODEL_PATH_TEXT, mtime)
except Exception as e:
    st.error("❌ Model found, but failed to load.")
    st.exception(e)
    st.stop()

# Class names from model
class_names = get_class_names_from_model(model)
out_dim = None
try:
    out_dim = int(model.output_shape[-1])
except Exception:
    out_dim = None

if not class_names:
    # Still works, but user-friendly warning
    class_names = [f"class_{i}" for i in range(out_dim or 0)]
    st.warning("⚠️ This model does not contain embedded class names. Showing class_0, class_1, ...")
else:
    st.success(f"✅ Model ready — {len(class_names)} classes loaded.")

# ---- Main: Predict
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("1) Upload a leaf image")
    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")

    if uploaded:
        pil = Image.open(uploaded)
        st.image(pil, caption="Uploaded image", use_container_width=True)

with right:
    st.subheader("2) Predict")
    st.write("When you are ready, click **Predict** to run the model.")

    if uploaded:
        size = model_input_size(model)

        if st.button("Predict", type="primary", use_container_width=True):
            with st.spinner("Predicting..."):
                batch = preprocess_image_0_255(pil, size)
                raw = model.predict(batch, verbose=0)
                probs = to_probs(raw)

            best_idx = int(np.argmax(probs))
            best_name = class_names[best_idx] if best_idx < len(class_names) else f"class_{best_idx}"
            best_conf = float(probs[best_idx])

            st.subheader("✅ Result")
            st.metric(label="Prediction", value=best_name)
            st.progress(min(max(best_conf, 0.0), 1.0), text=f"Confidence: {format_percent(best_conf)}")

            st.subheader("Top-5 predictions")
            top_k = min(5, len(probs))
            top_idx = np.argsort(probs)[::-1][:top_k]

            rows = []
            for i in top_idx:
                name = class_names[i] if i < len(class_names) else f"class_{i}"
                rows.append({"Class": name, "Confidence": float(probs[i])})

            # Nicely formatted table
            st.dataframe(
                rows,
                use_container_width=True,
                column_config={
                    "Confidence": st.column_config.NumberColumn(
                        "Confidence",
                        format="%.4f",
                    )
                },
                hide_index=True,
            )

            st.caption("Note: Confidence is the model’s probability score and may be overconfident on unfamiliar photos.")
    else:
        st.info("Upload an image to enable prediction.")
