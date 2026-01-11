import json
import io
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st


# -----------------------------
# ✅ Custom layer (REQUIRED to load your model)
# The model config contains: registered_name = "meta>ClassNamesLayer"
# -----------------------------
@tf.keras.utils.register_keras_serializable(package="meta")
class ClassNamesLayer(tf.keras.layers.Layer):
    def __init__(self, class_names=None, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.class_names = list(class_names) if class_names is not None else []

    def call(self, inputs):
        return inputs  # pass-through layer

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"class_names": self.class_names})
        return cfg


# -----------------------------
# Page setup (same look, two-panel layout)
# -----------------------------
st.set_page_config(
    page_title="Plant Disease identification with AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# UI tweaks (left panel red + title sizing + rename uploader button)
# -----------------------------
st.markdown(
    """
<style>
/* --- Make the LEFT panel red (the first column of the main horizontal block) --- */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child {
    background: #b00020;              /* red */
    padding: 1.25rem 1rem;
    border-radius: 14px;
}

/* Make text inside the left panel white for readability */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child * {
    color: #ffffff !important;
}

/* Slightly nicer expander header in the left panel */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child summary {
    background: rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 0.55rem 0.75rem;
}

/* --- Rename the "Browse files" button text inside the uploader --- */
div[data-testid="stFileUploader"] button {
    font-size: 0px !important;        /* hide original "Browse files" */
}
div[data-testid="stFileUploader"] button::after {
    content: "Take/Upload Photo";     /* new label */
    font-size: 14px;
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Hidden paths (NO sidebar settings)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR / "models" / "best_model.keras").resolve()
CLASSES_PATH = (BASE_DIR / "class_names.json").resolve()

# -----------------------------
# Your rules (thresholds)
# -----------------------------
CONFIDENCE_THRESHOLD = 0.50

# NEW: reject obvious non-leaf / bad quality
LEAF_RATIO_MIN = 0.05        # how much of the image looks like vegetation-ish colors
BRIGHTNESS_MIN = 0.12        # too dark -> reject
BLUR_VAR_MIN = 60.0          # too blurry -> reject (adjust if needed)


# -----------------------------
# Helper functions
# -----------------------------
def load_class_names(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or not names:
        raise ValueError("class_names.json must be a non-empty JSON list.")
    return names


def extract_embedded_class_names(m: tf.keras.Model) -> list[str] | None:
    """Find ClassNamesLayer anywhere in the model (even nested) and return its class_names."""
    def walk(layer):
        if isinstance(layer, ClassNamesLayer) and getattr(layer, "class_names", None):
            return layer.class_names
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                out = walk(sub)
                if out:
                    return out
        return None

    for layer in getattr(m, "layers", []):
        out = walk(layer)
        if out:
            return out
    return None


@st.cache_resource
def load_model_cached(model_path: str, mtime: float):
    # ✅ IMPORTANT: provide custom_objects so Keras can deserialize ClassNamesLayer
    custom_objects = {"ClassNamesLayer": ClassNamesLayer}

    # Cache the model so it doesn't reload on every Streamlit rerun
    try:
        return tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False,   # Keras 3
        )
    except TypeError:
        # TensorFlow 2.15 / older Keras: safe_mode may not exist
        return tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False,
        )


def model_has_rescaling_layer(model: tf.keras.Model) -> bool:
    """Return True if the model (even nested) contains a Rescaling layer."""
    def _has(layer) -> bool:
        if layer.__class__.__name__.lower() == "rescaling":
            return True
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                if _has(sub):
                    return True
        return False
    return _has(model)


def preprocess(img: Image.Image, model: tf.keras.Model) -> np.ndarray:
    """
    Convert PIL image -> NumPy batch (1, H, W, 3)
    Resize to model input size.
    Do NOT divide by 255 if model already contains Rescaling(1/255).
    """
    img = img.convert("RGB")

    in_shape = getattr(model, "input_shape", None)  # (None, 256, 256, 3)
    if isinstance(in_shape, tuple) and len(in_shape) == 4:
        target_h, target_w = in_shape[1], in_shape[2]
        if target_h is not None and target_w is not None:
            img = img.resize((target_w, target_h), Image.BILINEAR)

    x = np.array(img)              # (H, W, 3)
    x = np.expand_dims(x, 0)       # (1, H, W, 3)
    x = x.astype("float32")

    if not model_has_rescaling_layer(model):
        x = x / 255.0

    return x


def to_probabilities(pred_vector: np.ndarray) -> np.ndarray:
    """Ensure the output behaves like probabilities. If not, apply softmax."""
    pred_vector = np.asarray(pred_vector).astype("float32")
    s = float(pred_vector.sum())
    if not (0.98 <= s <= 1.02) or (pred_vector.min() < 0):
        pred_vector = tf.nn.softmax(pred_vector).numpy()
    return pred_vector


def image_quality_and_leafness(img: Image.Image) -> dict:
    """
    Fast heuristics to reject obvious non-leaf images:
    - brightness (too dark)
    - blur (Laplacian variance)
    - leaf_ratio: % pixels in vegetation-ish hue range (HSV)
    """
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    brightness = float(arr.mean() / 255.0)

    # Blur score: Laplacian variance (simple 4-neighbor Laplacian)
    gray = arr.mean(axis=2)
    up = np.roll(gray, -1, axis=0)
    down = np.roll(gray, 1, axis=0)
    left = np.roll(gray, -1, axis=1)
    right = np.roll(gray, 1, axis=1)
    lap = (up + down + left + right) - 4.0 * gray
    blur_var = float(lap.var())

    # Leaf ratio via HSV (vectorized)
    rgb = arr / 255.0
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    h = np.zeros_like(maxc)
    mask = delta > 1e-6

    # hue calc
    idx = mask & (maxc == r)
    h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6.0
    idx = mask & (maxc == g)
    h[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2.0
    idx = mask & (maxc == b)
    h[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4.0
    h = (h / 6.0) % 1.0  # 0..1

    s = np.zeros_like(maxc)
    idx2 = maxc > 1e-6
    s[idx2] = delta[idx2] / maxc[idx2]
    v = maxc

    # vegetation-ish hues: yellow->green (tolerant)
    leaf_mask = (h >= 0.12) & (h <= 0.50) & (s >= 0.15) & (v >= 0.15)
    leaf_ratio = float(leaf_mask.mean())

    return {"brightness": brightness, "blur_var": blur_var, "leaf_ratio": leaf_ratio}


# -----------------------------
# Load model + class names (hidden)
# -----------------------------
model = None
class_names = None

model_error = None
classes_error = None

if not MODEL_PATH.exists():
    model_error = "Model file not found ❗"

if model_error is None:
    try:
        model = load_model_cached(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    except Exception as e:
        model_error = f"Model found, but failed to load ❌\n\n{e}"

# ✅ Prefer embedded class names from the model; fallback to class_names.json if needed
if model is not None:
    class_names = extract_embedded_class_names(model)

if class_names is None:
    if CLASSES_PATH.exists():
        try:
            class_names = load_class_names(CLASSES_PATH)
        except Exception as e:
            classes_error = f"class_names.json found, but failed to load ❌\n\n{e}"
    else:
        classes_error = "No embedded class names in model AND class_names.json not found ❗"


# -----------------------------
# TWO "PAGES" (LEFT / RIGHT)
# -----------------------------
left, right = st.columns([1, 3], gap="large")


# -----------------------------
# LEFT: User Manual (collapsible)
# -----------------------------
with left:
    with st.expander("📘 User Manual", expanded=False):
        st.markdown(
            """
**How to take a good photo (important):**
- Use **bright natural light** (avoid very dark photos).
- Keep the leaf **in focus** (**no blur**).
- Capture **one leaf clearly** (fill most of the frame).
- Use a **plain background** if possible.
- Avoid strong **shadows**, **reflections**, and **filters**.
- Don’t crop too tightly — include the **full infected area**.

**How to use the app:**
1. Click **Take/Upload Photo** and upload a leaf image (**PNG / JPG / JPEG**).
2. Wait a moment for the prediction.
3. Read the **predicted class** and **confidence**.
            """
        )


# -----------------------------
# RIGHT: Title + Upload + Predict
# -----------------------------
with right:
    st.markdown(
        """
<div style="display:flex; align-items:flex-start; gap:0.75rem;">
  <div style="font-size:2.6rem; line-height:1;"></div>
  <div style="font-size:2.6rem; font-weight:700; line-height:1.08;">
    Plant Disease identification with AI 🌿
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.caption(
        "Upload a plant leaf image and this app will identify the plant disease using our trained artificial intelligence model "
        "(TensorFlow/Keras)."
    )

    st.divider()

    if model_error:
        st.error("Model is not loaded. Please contact the app owner.")
        st.caption(model_error)
        st.stop()

    if classes_error:
        st.error("Class names are not loaded. Please contact the app owner.")
        st.caption(classes_error)
        st.stop()

    uploaded = st.file_uploader("Take/Upload Photo", type=["png", "jpg", "jpeg"], key="uploader")

    if uploaded is None:
        st.info(
            "Upload photos and get the result.\n"
            "For best results, follow the User Manual on the left.\n"
            "For any issues, please contact the app owner at .sherzadzabihullah@yahoo.com"
        )
        st.stop()

    img_bytes = uploaded.getvalue()
    img_hash = hashlib.md5(img_bytes).hexdigest()

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    st.image(img, caption=f"Uploaded image (hash: {img_hash[:8]})", use_container_width=True)

    if st.button("Reset / Clear image"):
        st.session_state["last_hash"] = None
        st.session_state["last_pred"] = None
        st.session_state["last_probs"] = None
        st.session_state["last_is_confident"] = None
        st.rerun()

    # -----------------------------
    # Reject obvious non-leaf / bad quality BEFORE prediction
    # -----------------------------
    q = image_quality_and_leafness(img)

    if q["brightness"] < BRIGHTNESS_MIN or q["blur_var"] < BLUR_VAR_MIN:
        st.warning("⚠️ The image is blur or low quality, please upload another photo and try again.")
        st.stop()

    if q["leaf_ratio"] < LEAF_RATIO_MIN:
        st.warning("⚠️ This does not look like a plant leaf. Please upload a clear leaf photo and try again.")
        st.stop()

    # -----------------------------
    # Predict
    # -----------------------------
    x = preprocess(img, model)

    if st.session_state.get("last_hash") != img_hash or st.session_state.get("last_probs") is None:
        preds = model.predict(x, verbose=0)

        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        preds = np.asarray(preds)
        if preds.ndim == 2:
            preds = preds[0]

        probs = to_probabilities(preds)
        pred_id = int(np.argmax(probs))

        if pred_id >= len(class_names):
            st.error(
                f"Prediction index {pred_id} is outside class_names list (length {len(class_names)}). "
                "Fix: embedded class names (or class_names.json) must match model output order."
            )
            st.stop()

        best_conf = float(probs[pred_id])
        is_confident = best_conf >= CONFIDENCE_THRESHOLD

        st.session_state["last_hash"] = img_hash
        st.session_state["last_probs"] = probs
        st.session_state["last_pred"] = pred_id
        st.session_state["last_is_confident"] = is_confident

    probs = st.session_state["last_probs"]
    pred_id = int(st.session_state["last_pred"])
    confidence = float(probs[pred_id])

    # show prediction only if best >= 50%
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ The image is blur or low quality, please upload another photo and try again.")
        st.stop()

    pred_label = class_names[pred_id]

    st.success(f"✅ Predicted class: **{pred_label}**")
    st.write(f"Confidence: **{confidence:.2%}**")

    st.subheader("3) Top predictions (≥ 50%)")
    idx_over = np.where(np.asarray(probs) >= CONFIDENCE_THRESHOLD)[0]
    idx_over = idx_over[np.argsort(np.asarray(probs)[idx_over])[::-1]]

    for rank, i in enumerate(idx_over, start=1):
        st.write(f"{rank}. {class_names[int(i)]} — {float(probs[int(i)]):.2%}")

    st.caption("Tip: If predictions look wrong, try a brighter/sharper photo with a plain background.")
