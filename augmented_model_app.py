import io
import json
from pathlib import Path
from collections import deque

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import streamlit as st


# -----------------------------
# Paths (edit if your repo differs)
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "image_classification_model.keras"
CLASS_NAMES_PATH = APP_DIR / "models" / "class_names.json"


# -----------------------------
# Page setup (two-panel layout)
# -----------------------------
st.set_page_config(
    page_title="Plant Disease identification with AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# UI tweaks (left panel red + nicer upload button)
# -----------------------------
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
div[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.10) !important;
    border: 1px dashed rgba(255,255,255,0.45) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


<<<<<<< HEAD:augmented_model_app.py
# ✅ Put your augmented model here (Linux compatible .keras)
MODEL_PATH = (BASE_DIR / "models" / "augmented_model_linux.keras").resolve()
CLASSES_PATH = (BASE_DIR / "classes.json").resolve()
=======
# ============================================================
# Background removal (HSV) — FIXED to keep bright/glare areas
# ============================================================
def _fill_holes(fg_mask: np.ndarray) -> np.ndarray:
    """Fill holes in a boolean foreground mask by flood-filling background from borders."""
    fg = fg_mask.astype(bool)
    h, w = fg.shape
    bg = ~fg
>>>>>>> parent of a1ec998d (restore):app.py

    visited = np.zeros((h, w), dtype=bool)
    q = deque()

    # Seed BFS from border background pixels
    for x in range(w):
        if bg[0, x] and not visited[0, x]:
            visited[0, x] = True
            q.append((0, x))
        if bg[h - 1, x] and not visited[h - 1, x]:
            visited[h - 1, x] = True
            q.append((h - 1, x))

    for y in range(h):
        if bg[y, 0] and not visited[y, 0]:
            visited[y, 0] = True
            q.append((y, 0))
        if bg[y, w - 1] and not visited[y, w - 1]:
            visited[y, w - 1] = True
            q.append((y, w - 1))

    # Flood fill
    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and bg[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))

    holes = bg & ~visited
    return fg | holes


<<<<<<< HEAD:augmented_model_app.py
# -----------------------------
# Helpers: loading
# -----------------------------
def load_class_names(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or not names:
        raise ValueError("classes.json must be a non-empty JSON list.")
    return names


@st.cache_resource
def load_model_cached(model_path: str, mtime: float):
    # Some TF/Keras versions accept safe_mode, others not
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


# -----------------------------
# Helpers: preprocessing / postprocessing
# -----------------------------
def preprocess(img: Image.Image, model: tf.keras.Model) -> np.ndarray:
=======
def remove_bg_hsv_keep_bright(
    img: Image.Image,
    *,
    h_min=25,
    h_max=120,
    s_min=12,   # ✅ very low so bright glare (low S) is kept
    v_min=25,   # ✅ allow bright pixels (no V max!)
    max_dim=900
):
>>>>>>> parent of a1ec998d (restore):app.py
    """
    HSV leaf mask that does NOT remove bright glare areas.
    Returns: rgba_image, kept_ratio (0..1)
    """
    img = img.convert("RGB")

    # Speed: compute mask on a resized version, then scale back up
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        img_small = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    else:
        img_small = img

    rgb = np.array(img_small)
    hsv = np.array(Image.fromarray(rgb).convert("HSV"))

    hh = hsv[..., 0].astype(np.int16)  # 0..255
    ss = hsv[..., 1].astype(np.int16)  # 0..255
    vv = hsv[..., 2].astype(np.int16)  # 0..255

    # ✅ Key fix: keep low-saturation bright pixels (glare) by using low s_min
    # ✅ No rule like "V must be < ..." that would kill bright regions
    green = (hh >= h_min) & (hh <= h_max) & (ss >= s_min) & (vv >= v_min)

    mask_small = _fill_holes(green)
    kept_ratio = float(mask_small.mean())

    # If mask is too small, don't remove background (safe fallback)
    if kept_ratio < 0.20:
        return img.convert("RGBA"), 1.0

    # Resize mask back to original size (nearest = crisp edges)
    mask_small_u8 = (mask_small.astype(np.uint8) * 255)
    mask_img = Image.fromarray(mask_small_u8, mode="L").resize((w, h), Image.NEAREST)

    rgba = img.convert("RGBA")
    rgba_np = np.array(rgba)
    rgba_np[..., 3] = np.array(mask_img)
    return Image.fromarray(rgba_np), kept_ratio


def rgba_to_rgb_on_white(rgba: Image.Image) -> Image.Image:
    """Composite RGBA onto white background (model usually expects RGB)."""
    rgba = rgba.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    comp = Image.alpha_composite(white, rgba)
    return comp.convert("RGB")


# -----------------------------
# Model + classes loaders
# -----------------------------
@st.cache_resource
def load_model_cached(path: str):
    # compile=False makes loading more robust for Streamlit deployments
    return tf.keras.models.load_model(path, compile=False, safe_mode=False)


@st.cache_data
def load_class_names(path: str):
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # allow list or dict formats
    if isinstance(data, dict) and "class_names" in data:
        return data["class_names"]
    return data if isinstance(data, list) else None


def model_input_size(model) -> int:
    # Expected shape: (None, H, W, C)
    try:
        h = model.input_shape[1]
        w = model.input_shape[2]
        if isinstance(h, int) and isinstance(w, int) and h == w:
            return h
    except Exception:
        pass
    return 256


def model_has_rescaling(model) -> bool:
    # If model already includes Rescaling, don't divide by 255 again.
    for layer in getattr(model, "layers", []):
        if "Rescaling" in layer.__class__.__name__:
            return True
    return False


def softmax_if_needed(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).astype(np.float32)
    s = float(np.sum(x))
    # If it doesn't look like probabilities, apply softmax
    if not (0.98 <= s <= 1.02) or np.any(x < 0) or np.any(x > 1):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)
    return x


# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1, 1.35], gap="large")

with left:
    st.markdown("## 🌿 Plant Disease identification with AI")
    st.markdown(
        "Upload a clear leaf photo. The app will remove the background and predict the disease."
    )

    uploaded = st.file_uploader(
        "📷 Upload leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
    )

    st.markdown("### Preprocessing")
    method = st.selectbox(
        "Background removal method",
        ["HSV (recommended)", "None"],
        index=0,
    )

    with st.expander("Advanced (optional)", expanded=False):
        st.caption("Use these only if you really need them.")
        auto_crop = st.checkbox("Auto-crop to leaf region (experimental)", value=False)
        show_debug = st.checkbox("Show debug info", value=False)

    st.markdown("---")
    st.markdown("### Photo tips")
    st.markdown(
        "- Avoid strong flash/glare (but the new mask is more robust)\n"
        "- Hold the leaf flat and close to the camera\n"
        "- Use a simple background (paper/table)\n"
        "- Keep the leaf in focus (sharp image)"
    )

with right:
    st.markdown("## Results")

    if not uploaded:
        st.info("Upload an image to get a prediction.")
        st.stop()

    # Load image safely
    try:
        img = Image.open(io.BytesIO(uploaded.getvalue()))
        img = ImageOps.exif_transpose(img)  # fixes rotated phone photos
    except Exception as e:
        st.error(f"Could not open this image. Error: {e}")
        st.stop()

    # Apply background removal
    kept_ratio = None
    processed_preview = img.convert("RGB")

    if method == "HSV (recommended)":
        rgba, kept_ratio = remove_bg_hsv_keep_bright(img)
        processed_preview = rgba  # show RGBA preview
        model_img = rgba_to_rgb_on_white(rgba)
    else:
        model_img = img.convert("RGB")

    # Optional crop using alpha channel bounding box
    if method == "HSV (recommended)" and auto_crop:
        alpha = np.array(processed_preview.convert("RGBA"))[..., 3]
        ys, xs = np.where(alpha > 0)
        if len(xs) > 0 and len(ys) > 0:
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            pad = int(0.05 * max((x1 - x0 + 1), (y1 - y0 + 1)))
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(model_img.size[0] - 1, x1 + pad)
            y1 = min(model_img.size[1] - 1, y1 + pad)
            model_img = model_img.crop((x0, y0, x1 + 1, y1 + 1))
            processed_preview = processed_preview.crop((x0, y0, x1 + 1, y1 + 1))

    # Show images
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original**")
        st.image(img, use_container_width=True)
    with c2:
        st.markdown("**After preprocessing**")
        st.image(processed_preview, use_container_width=True)

    if kept_ratio is not None:
        st.caption(f"Method: hsv | Kept ratio: {kept_ratio*100:.2f}%")

    # Load model
    if not MODEL_PATH.exists():
        st.error(f"Model not found at: {MODEL_PATH}")
        st.stop()

    model = load_model_cached(str(MODEL_PATH))
    class_names = load_class_names(str(CLASS_NAMES_PATH))

    # Preprocess for model
    size = model_input_size(model)
    x = model_img.resize((size, size), Image.BILINEAR)
    x = np.array(x).astype(np.float32)

    if not model_has_rescaling(model):
        x /= 255.0

    x = np.expand_dims(x, axis=0)  # (1, H, W, 3)

    # Predict
    preds = model.predict(x, verbose=0)[0]
    probs = softmax_if_needed(preds)

    # Class names fallback
    if not class_names or len(class_names) != len(probs):
        class_names = [f"class_{i}" for i in range(len(probs))]

    top_idx = int(np.argmax(probs))
    top_name = class_names[top_idx]
    top_conf = float(probs[top_idx])

    st.markdown("### ✅ Prediction")
    st.markdown(f"**Predicted class:** {top_name}")
    st.markdown(f"**Confidence:** {top_conf*100:.2f}%")

    # Top-3 (>= 50%)
    st.markdown("### 3) Top predictions (≥ 50%)")
    top3 = np.argsort(probs)[::-1][:3]
    shown = 0
    for i in top3:
        if float(probs[i]) >= 0.50:
            st.write(f"- {class_names[int(i)]} — {float(probs[i])*100:.2f}%")
            shown += 1
    if shown == 0:
        st.write("- (No class ≥ 50%)")

    st.caption("Tip: If predictions look wrong, try a brighter/sharper photo, or enable Auto-crop (Advanced).")

    if "show_debug" in locals() and show_debug:
        st.markdown("---")
        st.markdown("### Debug")
        st.write("Model input size:", size)
        st.write("MODEL_PATH:", str(MODEL_PATH))
        st.write("CLASS_NAMES_PATH:", str(CLASS_NAMES_PATH))
        st.write("Model has Rescaling:", model_has_rescaling(model))
        st.write("Prob sum:", float(np.sum(probs)))
