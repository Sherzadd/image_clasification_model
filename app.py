from pathlib import Path

updated_code = r'''import json
import io
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import tensorflow as tf
import streamlit as st


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
# UI tweaks (left panel red + rename uploader button)
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

# -----------------------------
# Hidden paths (NO sidebar settings)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR / "models" / "image_classification_model_linux.keras").resolve()
CLASSES_PATH = (BASE_DIR / "class_names.json").resolve()

# -----------------------------
# Rules / thresholds
# -----------------------------
CONFIDENCE_THRESHOLD = 0.50

BRIGHTNESS_MIN = 0.12        # too dark -> reject
BLUR_VAR_MIN = 60.0          # too blurry -> reject

# Masking thresholds (loose, because we do NOT want to block predictions)
KEPT_RATIO_MIN = 0.015       # very small is okay; we mainly want a bbox

# ✅ NEW: "No leaf / unknown" gate
# If your mask keeps very little, it's probably NOT a leaf photo (documents, faces, objects, etc.)
MIN_LEAF_RATIO_HARD = 0.15   # below this => almost certainly not a leaf
MIN_LEAF_RATIO_SOFT = 0.25   # low leaf content (whole plant / clutter)
MIN_CONFIDENCE_SOFT = 0.85   # if also low confidence => reject

# ✅ NEW: robust whole-plant mode (patch voting)
PATCH_SIZE = 256
PATCH_STRIDE = 160
PATCH_LEAF_RATIO_MIN = 0.12  # per-patch leaf requirement in robust mode
PATCH_TOP_K = 10
PATCH_MAX_SIDE = 1024        # resize big photos down for speed


# -----------------------------
# Helpers: loading
# -----------------------------
def load_class_names(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or not names:
        raise ValueError("class_names.json must be a non-empty JSON list.")
    return names


@st.cache_resource
def load_model_cached(model_path: str, mtime: float):
    # Some environments support safe_mode, others not
    try:
        return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    except TypeError:
        return tf.keras.models.load_model(model_path, compile=False)


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

    x = np.array(img, dtype=np.float32)  # (H,W,3)
    x = np.expand_dims(x, 0)             # (1,H,W,3)

    if not model_has_rescaling_layer(model):
        x = x / 255.0

    return x


def to_probabilities(pred_vector: np.ndarray) -> np.ndarray:
    """Ensure output behaves like probabilities. If not, apply softmax."""
    pred_vector = np.asarray(pred_vector, dtype=np.float32)
    s = float(pred_vector.sum())
    if not (0.98 <= s <= 1.02) or (pred_vector.min() < 0.0) or (pred_vector.max() > 1.0):
        pred_vector = tf.nn.softmax(pred_vector).numpy()
    return pred_vector


def image_quality(img: Image.Image) -> dict:
    """Brightness + blur (Laplacian variance)."""
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


# -----------------------------
# Mask utilities (no OpenCV)
# -----------------------------
def _fill_holes(mask: np.ndarray, max_iters: int = 2000) -> np.ndarray:
    """
    Fill holes inside a binary mask using flood-fill from image borders
    on the inverse mask.
    """
    mask = mask.astype(bool)
    inv = ~mask

    reach = np.zeros_like(inv, dtype=bool)
    reach[0, :] = inv[0, :]
    reach[-1, :] = inv[-1, :]
    reach[:, 0] = inv[:, 0]
    reach[:, -1] = inv[:, -1]

    for _ in range(max_iters):
        nb = (
            np.roll(reach, 1, 0) | np.roll(reach, -1, 0) |
            np.roll(reach, 1, 1) | np.roll(reach, -1, 1)
        )
        new = reach | (nb & inv)
        if new.sum() == reach.sum():
            reach = new
            break
        reach = new

    holes = inv & ~reach
    return mask | holes


def _bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1


def _pad_bbox(bbox, W, H, pad_frac: float = 0.06):
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    pad = int(pad_frac * max(bw, bh))

    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad)
    y1 = min(H - 1, y1 + pad)
    return x0, y0, x1, y1


def _apply_white_bg(img: Image.Image, mask: np.ndarray) -> Image.Image:
    """Preview-only: show masked image with white background."""
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    white = Image.new("RGB", img.size, (255, 255, 255))
    return Image.composite(img, white, mask_img)


# -----------------------------
# Mask methods
# -----------------------------
def mask_by_corners(img: Image.Image, patch: int = 24, percentile: float = 99.5, extra_margin: float = 0.05):
    """
    Estimate background color from 4 corners, keep pixels far from it.
    Good when corners are true background.
    """
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    H, W, _ = arr.shape

    p = int(min(patch, H // 3, W // 3))
    if p < 4:
        return None

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

    raw = dist > thr

    m = Image.fromarray((raw.astype(np.uint8) * 255), mode="L")
    m = m.filter(ImageFilter.MedianFilter(size=5))
    m = m.filter(ImageFilter.GaussianBlur(radius=1.0))
    mask = (np.array(m) > 40)

    mask = _fill_holes(mask)
    kept_ratio = float(mask.mean())

    return {"mask": mask, "kept_ratio": kept_ratio, "method": "corners"}


def mask_by_hsv(img: Image.Image):
    """
    Fallback: vegetation-like colors + green-dominance.
    IMPORTANT: this is only used to get a bbox; prediction uses original crop.
    """
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    h = np.zeros_like(maxc)
    m = delta > 1e-6

    idx = m & (maxc == r)
    h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6.0
    idx = m & (maxc == g)
    h[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2.0
    idx = m & (maxc == b)
    h[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4.0
    h = (h / 6.0) % 1.0

    s = np.zeros_like(maxc)
    idx2 = maxc > 1e-6
    s[idx2] = delta[idx2] / maxc[idx2]
    v = maxc

    # relaxed vegetation-ish thresholds
    hsv_leaf = (h >= 0.08) & (h <= 0.58) & (s >= 0.10) & (v >= 0.10)

    # also accept "green dominance" pixels
    green_dom = (g > r * 1.02) & (g > b * 1.02) & (v >= 0.08)

    raw = hsv_leaf | green_dom

    m_img = Image.fromarray((raw.astype(np.uint8) * 255), mode="L")
    m_img = m_img.filter(ImageFilter.MedianFilter(size=5))
    m_img = m_img.filter(ImageFilter.GaussianBlur(radius=1.0))
    mask = (np.array(m_img) > 40)

    mask = _fill_holes(mask)
    kept_ratio = float(mask.mean())

    return {"mask": mask, "kept_ratio": kept_ratio, "method": "hsv"}


def mask_leaf_for_prediction(img: Image.Image):
    """
    Returns:
      pred_img  -> ORIGINAL crop (keeps symptoms!)
      preview_img -> masked preview (optional UI)
      info
    """
    W, H = img.size

    # 1) corners
    out = mask_by_corners(img)
    if out is not None and out["kept_ratio"] >= KEPT_RATIO_MIN:
        bbox = _bbox_from_mask(out["mask"])
        if bbox is not None:
            bbox = _pad_bbox(bbox, W, H)
            x0, y0, x1, y1 = bbox
            pred_img = img.crop((x0, y0, x1 + 1, y1 + 1))  # ORIGINAL crop ✅
            preview = _apply_white_bg(img, out["mask"]).crop((x0, y0, x1 + 1, y1 + 1))
            out["bbox"] = bbox
            return pred_img, preview, out

    # 2) hsv fallback
    out2 = mask_by_hsv(img)
    if out2 is not None and out2["kept_ratio"] >= KEPT_RATIO_MIN:
        bbox = _bbox_from_mask(out2["mask"])
        if bbox is not None:
            bbox = _pad_bbox(bbox, W, H)
            x0, y0, x1, y1 = bbox
            pred_img = img.crop((x0, y0, x1 + 1, y1 + 1))  # ORIGINAL crop ✅
            preview = _apply_white_bg(img, out2["mask"]).crop((x0, y0, x1 + 1, y1 + 1))
            out2["bbox"] = bbox
            return pred_img, preview, out2

    # 3) final fallback: no masking, no block
    return img, img, {"kept_ratio": 1.0, "method": "none", "bbox": None}


# -----------------------------
# Robust mode: patch voting (whole plant / clutter)
# -----------------------------
def _resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return img.resize((nw, nh), Image.BILINEAR)


def _iter_patches(img: Image.Image, patch: int, stride: int):
    w, h = img.size
    if w < patch or h < patch:
        # If the image is smaller, just yield the image itself
        yield (0, 0, img)
        return

    for y in range(0, h - patch + 1, stride):
        for x in range(0, w - patch + 1, stride):
            yield (x, y, img.crop((x, y, x + patch, y + patch)))


def predict_patch_voting(full_img: Image.Image, model: tf.keras.Model):
    """
    Split full image into patches, keep only patches that contain enough leaf pixels,
    predict per patch, then aggregate (top-k mean).
    Returns: probs, debug_info
    """
    full_img = _resize_max_side(full_img, PATCH_MAX_SIDE)

    patches = []
    patch_meta = []

    total = 0
    kept = 0

    for x0, y0, pimg in _iter_patches(full_img, PATCH_SIZE, PATCH_STRIDE):
        total += 1

        # Use fast mask just to decide if patch contains leaf
        m1 = mask_by_corners(pimg)
        info = m1 if (m1 is not None and m1["kept_ratio"] >= KEPT_RATIO_MIN) else mask_by_hsv(pimg)
        kr = float(info["kept_ratio"]) if info is not None else 0.0

        if kr >= PATCH_LEAF_RATIO_MIN:
            patches.append(pimg)
            patch_meta.append((x0, y0, kr))
            kept += 1

    if kept == 0:
        return None, {"total_patches": total, "kept_patches": 0}

    # Batch predict for speed
    batch = np.concatenate([preprocess(p, model) for p in patches], axis=0)
    preds = model.predict(batch, verbose=0)

    preds = np.asarray(preds)
    if preds.ndim == 1:
        preds = preds[None, :]

    probs_all = np.stack([to_probabilities(v) for v in preds], axis=0)  # (N,C)

    # Pick top-k patches by their max confidence
    conf = probs_all.max(axis=1)
    k = int(min(PATCH_TOP_K, probs_all.shape[0]))
    idx = np.argsort(conf)[-k:]
    probs = probs_all[idx].mean(axis=0)

    dbg = {
        "total_patches": total,
        "kept_patches": kept,
        "top_k_used": k,
        "avg_leaf_ratio_kept": float(np.mean([m[2] for m in patch_meta])) if patch_meta else 0.0,
    }
    return probs, dbg


# -----------------------------
# Load model + class names (hidden)
# -----------------------------
model_error = None
classes_error = None

if not MODEL_PATH.exists():
    model_error = "Model file not found ❗"

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
        class_names = load_class_names(CLASSES_PATH)
    except Exception as e:
        classes_error = f"class_names.json found, but failed to load ❌\n\n{e}"


# -----------------------------
# Layout: LEFT / RIGHT
# -----------------------------
left, right = st.columns([1, 3], gap="large")

with left:
    with st.expander("📘 User Manual", expanded=False):
        st.markdown(
            """
**How to take a good photo (important):**
- Use **bright natural light** (avoid very dark photos).
- Keep the leaf **in focus** (**no blur**).
- Capture **one leaf clearly** (fill most of the frame).
- A plain background helps, but **is NOT required**.

**Whole plant photos (recommended):**
- If you upload the **whole plant**, enable **Robust mode (patch voting)**.
- The app will scan many patches and focus on leaf/symptom regions.

**How the app works:**
- The app tries to **find the leaf area** and crops the image.
- The model predicts on the **original cropped leaf** (symptoms are preserved).
- If no leaf is detected, the app will ask for a clearer leaf photo.
            """
        )

with right:
    st.markdown(
        """
<div style="display:flex; align-items:flex-start; gap:0.75rem;">
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

    robust_mode = st.toggle("Robust mode (whole plant / clutter): patch voting", value=False)

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

    # ✅ EXIF-safe loading (phone vs computer consistency)
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")

    st.image(img, caption=f"Uploaded image (hash: {img_hash[:8]})", use_container_width=True)

    if st.button("Reset / Clear image"):
        for k in ["last_key", "last_pred", "last_probs", "last_debug"]:
            st.session_state.pop(k, None)
        st.rerun()

    # Global quality check (fast reject for extremely dark/blurry images)
    q_global = image_quality(img)
    if q_global["brightness"] < BRIGHTNESS_MIN * 0.75:
        st.warning("⚠️ The photo is too dark. Please take a brighter photo of the leaf.")
        st.stop()

    # -----------------------------
    # Prediction (two modes)
    # -----------------------------
    mode_key = "robust" if robust_mode else "single"
    cache_key = f"{img_hash}:{mode_key}"

    if st.session_state.get("last_key") != cache_key or st.session_state.get("last_probs") is None:
        if robust_mode:
            probs, dbg = predict_patch_voting(img, model)

            if probs is None:
                st.error("❌ No leaf detected. Please upload a clear leaf photo (close-up).")
                st.caption("Tip: Make sure the leaf is visible and fills more of the frame.")
                st.stop()

            st.session_state["last_debug"] = dbg

        else:
            # Single-leaf crop mode (fast)
            pred_img, preview_img, mask_info = mask_leaf_for_prediction(img)

            with st.expander("🧪 Show leaf crop / masking preview (used to locate the leaf)", expanded=False):
                st.image(preview_img, use_container_width=True)
                st.caption(f"Method: {mask_info['method']} | Kept ratio: {mask_info['kept_ratio']:.2%}")
                if mask_info["method"] == "none":
                    st.info("Masking wasn’t reliable here — using the original image for prediction (no crop).")

            leaf_ratio = float(mask_info.get("kept_ratio", 1.0))

            # ✅ NEW: reject obvious non-leaf images
            if leaf_ratio < MIN_LEAF_RATIO_HARD:
                st.error("❌ No leaf detected. Please upload a clear leaf photo (close-up).")
                st.caption(f"Leaf area detected: {leaf_ratio:.2%} (too low).")
                st.stop()

            # Quality checks on the image we actually feed to the model
            q = image_quality(pred_img)
            if q["brightness"] < BRIGHTNESS_MIN or q["blur_var"] < BLUR_VAR_MIN:
                st.warning("⚠️ The image is blur or low quality, please upload another photo and try again.")
                st.stop()

            x = preprocess(pred_img, model)
            preds = model.predict(x, verbose=0)

            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            preds = np.asarray(preds)
            if preds.ndim == 2:
                preds = preds[0]

            probs = to_probabilities(preds)
            st.session_state["last_debug"] = {"method": mask_info.get("method"), "kept_ratio": leaf_ratio}

        pred_id = int(np.argmax(probs))

        if pred_id >= len(class_names):
            st.error(
                f"Prediction index {pred_id} is outside class_names list (length {len(class_names)}). "
                "Fix: class_names.json must match the model output order."
            )
            st.stop()

        st.session_state["last_key"] = cache_key
        st.session_state["last_probs"] = probs
        st.session_state["last_pred"] = pred_id

    probs = st.session_state["last_probs"]
    pred_id = int(st.session_state["last_pred"])
    confidence = float(probs[pred_id])

    # Soft "unknown" gate (helps avoid confident wrong answers on weird inputs)
    if not robust_mode:
        dbg = st.session_state.get("last_debug", {})
        leaf_ratio = float(dbg.get("kept_ratio", 1.0))
        if leaf_ratio < MIN_LEAF_RATIO_SOFT and confidence < MIN_CONFIDENCE_SOFT:
            st.warning("⚠️ Leaf is too small/unclear for reliable prediction. Try a closer photo of one leaf.")
            st.caption(f"Leaf area detected: {leaf_ratio:.2%} | Confidence: {confidence:.2%}")
            st.stop()

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Low confidence. Please upload a clearer leaf photo and try again.")
        st.stop()

    pred_label = class_names[pred_id]

    st.success(f"✅ Predicted class: **{pred_label}**")
    st.write(f"Confidence: **{confidence:.2%}**")

    # Debug info for robust mode
    if robust_mode:
        dbg = st.session_state.get("last_debug", {})
        with st.expander("🧩 Robust mode details (patch voting)", expanded=False):
            st.write(f"Total patches scanned: {dbg.get('total_patches', '-')}")
            st.write(f"Patches used (leafy): {dbg.get('kept_patches', '-')}")
            st.write(f"Top‑K patches averaged: {dbg.get('top_k_used', '-')}")
            st.write(f"Avg leaf ratio (kept patches): {dbg.get('avg_leaf_ratio_kept', 0.0):.2%}")

    st.subheader("3) Top predictions (≥ 50%)")
    idx_over = np.where(np.asarray(probs) >= CONFIDENCE_THRESHOLD)[0]
    idx_over = idx_over[np.argsort(np.asarray(probs)[idx_over])[::-1]]

    for rank, i in enumerate(idx_over, start=1):
        st.write(f"{rank}. {class_names[int(i)]} — {float(probs[int(i)]):.2%}")

    st.caption("Tip: If predictions look wrong, try a brighter/sharper close-up photo of a single leaf.")
'''
# Write updated file
out_path = Path("/mnt/data/app_updated.py")
out_path.write_text(updated_code, encoding="utf-8")
str(out_path)

