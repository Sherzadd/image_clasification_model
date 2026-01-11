import json
import io
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st


# -----------------------------
# Page setup (same look, two-panel layout)
# -----------------------------
st.set_page_config(
    page_title="Plant Disease identification with AI",
    page_icon="🌿",
    layout="wide",
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

/* --- Rename the "Browse files" button text inside the uploader --- */
div[data-testid="stFileUploader"] button {
    font-size: 0px !important;        /* hide original "Browse files" */
}
div[data-testid="stFileUploader"] button::after {
    content: "Upload Photo";          /* new label */
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

# Try a few common locations so deployment doesn't break if you move the model into /models.
MODEL_CANDIDATES = [
    BASE_DIR / "image_classification_model_linux.keras",
    BASE_DIR / "image_classification_model.keras",
    BASE_DIR / "models" / "image_classification_model_linux.keras",
    BASE_DIR / "models" / "image_classification_model.keras",
    BASE_DIR.parent / "models" / "image_classification_model_linux.keras",
    BASE_DIR.parent / "models" / "image_classification_model.keras",
]
MODEL_PATH = next((p.resolve() for p in MODEL_CANDIDATES if p.exists()), MODEL_CANDIDATES[0].resolve())

CLASSES_CANDIDATES = [
    BASE_DIR / "class_names.json",
    BASE_DIR / "models" / "class_names.json",
    BASE_DIR.parent / "models" / "class_names.json",
]
CLASSES_PATH = next((p.resolve() for p in CLASSES_CANDIDATES if p.exists()), CLASSES_CANDIDATES[0].resolve())

# -----------------------------
# Your rules (thresholds)
# -----------------------------
CONFIDENCE_THRESHOLD = 0.50

# NEW: reject obvious non-leaf / bad quality
LEAF_RATIO_MIN = 0.12        # how much of the image looks like vegetation-ish colors
BRIGHTNESS_MIN = 0.12        # too dark -> reject
BLUR_VAR_MIN = 60.0          # too blurry -> reject (adjust if needed)


# -----------------------------
# Helper: Load class names
# -----------------------------
def load_class_names(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or not names:
        raise ValueError("class_names.json must be a non-empty JSON list.")
    return names


# -----------------------------
# Optional: Custom layer placeholder (safe even if your model doesn't use it)
# -----------------------------
class ClassNamesLayer(tf.keras.layers.Layer):
    """
    If your saved model includes a custom layer that stores class names,
    defining it here lets Streamlit Cloud load the model.
    This layer is a passthrough at inference time.
    """
    def __init__(self, class_names=None, **kwargs):
        super().__init__(**kwargs)
        self.class_names = class_names or []

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"class_names": self.class_names})
        return cfg

    def call(self, inputs):
        return inputs


@st.cache_resource
def load_model_cached(model_path: str, mtime: float):
    # Cache the model so it doesn't reload on every Streamlit rerun
    try:
        return tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={"ClassNamesLayer": ClassNamesLayer},
        )
    except TypeError:
        # In case this Keras build expects safe_mode
        return tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False,
            custom_objects={"ClassNamesLayer": ClassNamesLayer},
        )


def model_has_rescaling_layer(model: tf.keras.Model) -> bool:
    # Detect if the model already has a rescaling layer
    for layer in getattr(model, "layers", []):
        name = layer.__class__.__name__.lower()
        if "rescaling" in name:
            return True
    return False


def preprocess(img: Image.Image, model: tf.keras.Model) -> np.ndarray:
    """Resize image to model input shape and normalize if needed."""
    # Resize based on model input
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
    arr = np.asarray(img).astype("float32") / 255.0  # (H,W,3) 0..1
    if arr.ndim != 3 or arr.shape[2] != 3:
        return {"brightness": 0.0, "blur_var": 0.0, "leaf_ratio": 0.0}

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    brightness = float((0.2126 * r + 0.7152 * g + 0.0722 * b).mean())

    # Blur score: Laplacian variance (simple 4-neighbor Laplacian)
    gray = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype("float32")
    lap = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    blur_var = float(lap.var())

    # Approx HSV to detect "vegetation-ish" hues (fast)
    maxc = np.max(arr, axis=2)
    minc = np.min(arr, axis=2)
    delta = maxc - minc

    h = np.zeros_like(maxc)
    idx = delta > 1e-6

    # Hue calculation
    rc = np.zeros_like(maxc)
    gc = np.zeros_like(maxc)
    bc = np.zeros_like(maxc)
    rc[idx] = (maxc[idx] - r[idx]) / delta[idx]
    gc[idx] = (maxc[idx] - g[idx]) / delta[idx]
    bc[idx] = (maxc[idx] - b[idx]) / delta[idx]

    r_is_max = (r == maxc) & idx
    g_is_max = (g == maxc) & idx
    b_is_max = (b == maxc) & idx

    h[r_is_max] = (bc[r_is_max] - gc[r_is_max])
    h[g_is_max] = 2.0 + (rc[g_is_max] - bc[g_is_max])
    h[b_is_max] = 4.0 + (gc[b_is_max] - rc[b_is_max])
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
    model_error = f"Model file not found ❗ ({MODEL_PATH})"

if not CLASSES_PATH.exists():
    classes_error = f"class_names.json not found ❗ ({CLASSES_PATH})"

if model_error is None:
    try:
        # Use mtime so cache invalidates if you update the model file
        model = load_model_cached(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    except Exception as e:
        model_error = str(e)

if classes_error is None:
    try:
        class_names = load_class_names(CLASSES_PATH)
    except Exception as e:
        classes_error = str(e)


# -----------------------------
# Session state (avoid re-predicting on same image)
# -----------------------------
if "last_hash" not in st.session_state:
    st.session_state["last_hash"] = None
    st.session_state["last_pred"] = None
    st.session_state["last_probs"] = None
    st.session_state["last_is_confident"] = None


# -----------------------------
# Layout: left manual + right app
# -----------------------------
left, right = st.columns([1, 4], gap="large")

# -----------------------------
# LEFT: User Manual
# -----------------------------
with left:
    st.markdown("## 📘 User Manual")
    st.markdown(
        """
**How to use this app**
1. Take a photo or upload a leaf photo.
2. The model will analyze the image.
3. It will show a prediction only if confidence ≥ **50%**.

**Tips for best results**
- Take the photo in **good light** (bright, not too dark).
- Keep the leaf **in focus** (**no blur**).
- Use a **single leaf** close-up.
- Avoid busy background (soil, hands, other objects).
- Use **jpg/png** format.

**Common problems**
- If the image is **blurry** or **too dark**, the app will reject it.
- If the image is **not a leaf**, the app will reject it.

**Support**
If you have any issue, contact the app owner.
"""
    )

# -----------------------------
# RIGHT: Title + Upload + Predict
# -----------------------------
with right:
    st.markdown(
        """
<div style="display:flex; align-items:flex-start; gap:0.75rem;">
  <div style="font-size:2.15rem; line-height:1;"></div>
  <div style="font-size:2.15rem; font-weight:700; line-height:1.12; max-width: 900px; white-space: normal;">
    Plant Disease identification with AI 🌿
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()

    # Optional: quick debug panel (helps when deploying on Streamlit Cloud)
    with st.expander("Debug (for app owner)", expanded=False):
        st.write("Python:", f"{__import__('sys').version.split()[0]}")
        st.write("MODEL_PATH:", str(MODEL_PATH))
        st.write("MODEL exists:", bool(MODEL_PATH.exists()))
        if MODEL_PATH.exists():
            st.write("MODEL size (MB):", round(MODEL_PATH.stat().st_size / (1024**2), 2))

        st.write("CLASSES_PATH:", str(CLASSES_PATH))
        st.write("CLASSES exists:", bool(CLASSES_PATH.exists()))
        if CLASSES_PATH.exists():
            st.write("CLASSES size (KB):", round(CLASSES_PATH.stat().st_size / 1024, 2))

        st.write("Files in app folder:", sorted([p.name for p in BASE_DIR.glob("*")])[:50])
        models_dir = BASE_DIR / "models"
        if models_dir.exists():
            st.write("Files in /models:", sorted([p.name for p in models_dir.glob("*")])[:50])

    if model_error:
        st.error("Model is not loaded. Please contact the app owner.")
        st.caption(model_error)
        st.stop()

    if classes_error:
        st.error("Class names are not loaded. Please contact the app owner.")
        st.caption(classes_error)
        st.stop()

    tab_upload, tab_camera = st.tabs(["📤 Upload", "📷 Camera"])

    with tab_upload:
        uploaded_file = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"], key="uploader")

    with tab_camera:
        camera_file = st.camera_input("Take a photo", key="camera")

    # Prefer camera capture if provided; otherwise use uploaded file
    uploaded = camera_file if camera_file is not None else uploaded_file

    if uploaded is None:
        st.info(
            """Upload or take a photo and get the result.

For best results, follow the User Manual on the left.

For any issues, please contact the app owner at .sherzadzabihullah@yahoo.com"""
        )
        st.stop()

    img_bytes = uploaded.getvalue()
    img_hash = hashlib.md5(img_bytes).hexdigest()

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    st.image(img, caption=f"Input image (hash: {img_hash[:8]})", use_container_width=True)

    with st.expander("Show quality checks (brightness / blur / leafness)", expanded=False):
        q_preview = image_quality_and_leafness(img)
        st.write("Brightness:", round(float(q_preview["brightness"]), 4))
        st.write("Blur score (Laplacian variance):", round(float(q_preview["blur_var"]), 2))
        st.write("Leaf-like pixel ratio:", round(float(q_preview["leaf_ratio"]), 4))
        st.caption(
            "Rules: image must be bright enough and not blurry. "
            "If it doesn't look like a leaf (too little green/yellow-green), it will be rejected."
        )

    if st.button("Reset / Clear image"):
        st.session_state["last_hash"] = None
        st.session_state["last_pred"] = None
        st.session_state["last_probs"] = None
        st.session_state["last_is_confident"] = None
        st.rerun()

    # -----------------------------
    # NEW: Reject obvious non-leaf / bad quality BEFORE prediction
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
                "Fix: class_names.json must match the model output order."
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

    # YOUR RULE:
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
# -----------------------------