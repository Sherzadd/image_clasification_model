import json
import io
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import tensorflow as tf
import streamlit as st


# ============================================================
# Page setup (same look, two-panel layout)
# ============================================================
st.set_page_config(
    page_title="Plant Disease identification with AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# Custom layers (needed if your saved model contains them)
# Fixes Streamlit/Linux load errors like:
# "Cannot deserialize object of type RandomBackgroundReplace"
# ============================================================
try:
    # Keras 3 style
    from keras.saving import register_keras_serializable
except Exception:
    # TF/Keras 2 style fallback
    from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable()
class RandomBackgroundReplace(tf.keras.layers.Layer):
    """
    Deployment-safe stub:
    - Exists so the model can be deserialized on Streamlit Cloud.
    - Does NO changes during inference (training=False).
    """
    def __init__(self, p=0.35, **kwargs):
        super().__init__(**kwargs)
        self.p = float(p)

    def call(self, inputs, training=None):
        return inputs

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"p": self.p})
        return cfg


@register_keras_serializable()
class ClassNamesLayer(tf.keras.layers.Layer):
    """
    Compatibility layer for models that embedded class names inside the model.
    """
    def __init__(self, class_names=None, **kwargs):
        super().__init__(**kwargs)
        self.class_names = list(class_names) if class_names is not None else None

    def call(self, inputs, training=None):
        return inputs

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"class_names": self.class_names})
        return cfg


CUSTOM_OBJECTS = {
    "RandomBackgroundReplace": RandomBackgroundReplace,
    "ClassNamesLayer": ClassNamesLayer,
}


# ============================================================
# UI tweaks (left panel red + title sizing + rename uploader button)
# ============================================================
st.markdown(
    """
<style>
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child {
    background: #b00020;
    padding: 1.25rem 1rem;
    border-radius: 14px;
}

div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child,
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child * {
    color: #ffffff !important;
}

div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child summary {
    background: rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 0.55rem 0.75rem;
}

div[data-testid="stFileUploader"] button {
    font-size: 0px !important;
}
div[data-testid="stFileUploader"] button::after {
    content: "Take/Upload Photo";
    font-size: 14px;
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Hidden paths (NO sidebar settings)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = (BASE_DIR / "models").resolve()

MODEL_CANDIDATES = [
    MODELS_DIR / "image_classification_model_linux.keras",
    MODELS_DIR / "image_classification_model_linux1.keras",
    MODELS_DIR / "image_classification_model.keras",
]
MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), None)

CLASSES_PATH = (BASE_DIR / "class_names.json").resolve()


# ============================================================
# Your rules (thresholds)
# ============================================================
CONFIDENCE_THRESHOLD = 0.50

# Quality checks
BRIGHTNESS_MIN = 0.12
BLUR_VAR_MIN = 60.0

# Background masking sanity check
KEPT_RATIO_MIN = 0.05


# ============================================================
# Helper functions
# ============================================================
@st.cache_data(show_spinner=False)
def load_class_names_cached(path: str, mtime: float) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or not names:
        raise ValueError("class_names.json must be a non-empty JSON list.")
    return names


@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: str, mtime: float):
    """Cache the model and include custom_objects for Streamlit/Linux."""
    try:
        return tf.keras.models.load_model(
            model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
            safe_mode=False,
        )
    except TypeError:
        return tf.keras.models.load_model(
            model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
        )


def model_has_rescaling_layer(model: tf.keras.Model) -> bool:
    def _has(layer) -> bool:
        if layer.__class__.__name__.lower() == "rescaling":
            return True
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                if _has(sub):
                    return True
        return False
    return _has(model)


def get_model_input_hw(model: tf.keras.Model) -> tuple[int, int]:
    in_shape = getattr(model, "input_shape", None)
    if isinstance(in_shape, list) and len(in_shape) > 0:
        in_shape = in_shape[0]
    if isinstance(in_shape, tuple) and len(in_shape) == 4:
        h, w = in_shape[1], in_shape[2]
        if h is not None and w is not None:
            return int(h), int(w)
    return 256, 256


def preprocess(img: Image.Image, model: tf.keras.Model) -> np.ndarray:
    img = img.convert("RGB")
    target_h, target_w = get_model_input_hw(model)
    img = img.resize((target_w, target_h), Image.BILINEAR)

    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, 0)

    if not model_has_rescaling_layer(model):
        x = x / 255.0

    return x


def to_probabilities(pred_vector: np.ndarray) -> np.ndarray:
    pred_vector = np.asarray(pred_vector, dtype=np.float32)
    s = float(pred_vector.sum())
    if not (0.98 <= s <= 1.02) or (pred_vector.min() < 0.0) or (pred_vector.max() > 1.0):
        pred_vector = tf.nn.softmax(pred_vector).numpy()
    return pred_vector


def image_quality(img: Image.Image) -> dict:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    brightness = float(arr.mean() / 255.0)

    gray = arr.mean(axis=2)
    up = np.roll(gray, -1, axis=0)
    down = np.roll(gray, 1, axis=0)
    left = np.roll(gray, -1, axis=1)
    right = np.roll(gray, 1, axis=1)
    lap = (up + down + left + right) - 4.0 * gray
    blur_var = float(lap.var())

    return {"brightness": brightness, "blur_var": blur_var}


def mask_background_by_corners(
    img: Image.Image,
    patch: int = 24,
    percentile: float = 99.5,
    extra_margin: float = 0.05,
):
    """Mask background using corner patches; works for brown/dry leaves too."""
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    H, W, _ = arr.shape

    p = int(min(patch, H // 3, W // 3))
    if p < 4:
        return img, img, {"kept_ratio": 1.0, "threshold": 0.0}

    corners = np.concatenate([
        arr[:p, :p, :].reshape(-1, 3),
        arr[:p, W - p:, :].reshape(-1, 3),
        arr[H - p:, :p, :].reshape(-1, 3),
        arr[H - p:, W - p:, :].reshape(-1, 3),
    ], axis=0)

    bg = np.median(corners, axis=0)
    dist = np.linalg.norm(arr - bg[None, None, :], axis=2)

    corner_dist = np.linalg.norm(corners - bg[None, :], axis=1)
    thr = float(np.percentile(corner_dist, percentile) + extra_margin)

    mask = dist > thr

    m = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    m = m.filter(ImageFilter.MedianFilter(size=5))
    m = m.filter(ImageFilter.GaussianBlur(radius=1.0))
    mask_c = (np.array(m) > 40)

    kept_ratio = float(mask_c.mean())

    mask_img = Image.fromarray((mask_c.astype(np.uint8) * 255), mode="L")
    white = Image.new("RGB", img.size, (255, 255, 255))
    masked = Image.composite(img, white, mask_img)

    ys, xs = np.where(mask_c)
    masked_cropped = masked
    if len(xs) and len(ys):
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        pad = int(0.06 * max((x1 - x0 + 1), (y1 - y0 + 1)))
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(W - 1, x1 + pad)
        y1 = min(H - 1, y1 + pad)
        masked_cropped = masked.crop((x0, y0, x1 + 1, y1 + 1))

    return masked, masked_cropped, {"kept_ratio": kept_ratio, "threshold": thr}


# ============================================================
# Load model + class names (hidden)
# ============================================================
model_error = None
classes_error = None

if MODEL_PATH is None:
    model_error = "Model file not found ❗ (Expected inside /models)"

if not CLASSES_PATH.exists():
    classes_error = "class_names.json file not found ❗"

model = None
class_names = None

if model_error is None:
    try:
        model = load_model_cached(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    except Exception as e:
        model_error = f"Model found, but failed to load ❌\n\n{e}"

if classes_error is None:
    try:
        class_names = load_class_names_cached(str(CLASSES_PATH), CLASSES_PATH.stat().st_mtime)
    except Exception as e:
        classes_error = f"class_names.json found, but failed to load ❌\n\n{e}"


# ============================================================
# TWO "PAGES" (LEFT / RIGHT)
# ============================================================
left, right = st.columns([1, 3], gap="large")


# ============================================================
# LEFT: User Manual (collapsible)
# ============================================================
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
2. The app will **mask the background** automatically.
3. Read the **predicted class** and **confidence**.
            """
        )


# ============================================================
# RIGHT: Title + Upload + Predict
# ============================================================
with right:
    st.markdown(
        """
<div style="display:flex; align-items:flex-start; gap:0.75rem;">
  <div style="font-size:2.6rem; line-height:1;"></div>
  <div style="font-size:2.6rem; font-weight:700; line-height:1.08;">
    Plant Disease identification with AI🌿
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
            "Upload a photo and get the result.\n"
            "For best results, follow the User Manual on the left.\n"
            "For any issues, please contact the app owner."
        )
        st.stop()

    img_bytes = uploaded.getvalue()
    img_hash = hashlib.md5(img_bytes).hexdigest()

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    st.image(img, caption=f"Uploaded image (hash: {img_hash[:8]})", use_container_width=True)

    if st.button("Reset / Clear image"):
        for k in ["last_hash", "last_pred", "last_probs"]:
            st.session_state.pop(k, None)
        st.rerun()

    # ============================================================
    # Background masking (the model uses masked_cropped)
    # ============================================================
    masked, masked_cropped, mask_info = mask_background_by_corners(img)

    with st.expander("🧪 Show background-masked image (used for prediction)", expanded=False):
        st.image(masked_cropped, use_container_width=True)
        st.caption(f"Kept ratio: {mask_info['kept_ratio']:.2%}")

    if mask_info["kept_ratio"] < KEPT_RATIO_MIN:
        st.warning("⚠️ Could not isolate the leaf well. Please use a plain background and try again.")
        st.stop()

    # ============================================================
    # Quality checks (on the masked/cropped image)
    # ============================================================
    q = image_quality(masked_cropped)

    if q["brightness"] < BRIGHTNESS_MIN or q["blur_var"] < BLUR_VAR_MIN:
        st.warning("⚠️ The image is blur or low quality, please upload another photo and try again.")
        st.stop()

    # ============================================================
    # Predict (use masked_cropped)
    # ============================================================
    x = preprocess(masked_cropped, model)

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
                "Fix: class_names.json must match the model output order."
            )
            st.stop()

        st.session_state["last_hash"] = img_hash
        st.session_state["last_probs"] = probs
        st.session_state["last_pred"] = pred_id

    probs = st.session_state["last_probs"]
    pred_id = int(st.session_state["last_pred"])
    confidence = float(probs[pred_id])

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
