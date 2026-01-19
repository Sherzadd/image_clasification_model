import json
import io
import hashlib
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageFilter
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

MODEL_PATH = (BASE_DIR / "01_tomato_disease_model_linux.keras").resolve()
CLASSES_PATH = (BASE_DIR / "tomato_class_names.json").resolve()


# -----------------------------
# Rules / thresholds
# -----------------------------
CONFIDENCE_THRESHOLD = 0.50

BRIGHTNESS_MIN = 0.12        # too dark -> reject
BLUR_VAR_MIN = 60.0          # too blurry -> reject

# Masking thresholds (loose, because we do NOT want to block predictions)
KEPT_RATIO_MIN = 0.015       # very small is okay; we mainly want a bbox


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


def image_quality(img: Image.Image, mask_255: np.ndarray | None = None) -> dict:
    """Brightness + blur (Laplacian variance). If a mask is provided, compute metrics on masked pixels only."""
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)

    if mask_255 is not None and getattr(mask_255, "size", 0) and bool(np.any(mask_255 > 0)):
        m = (mask_255 > 0).astype(np.float32)
        denom = float(m.sum())
        if denom >= 10:
            brightness = float((gray * m).sum() / denom / 255.0)
            lap = cv2.Laplacian(gray, cv2.CV_32F)
            blur_var = float(lap[m > 0].var())
            return {"brightness": brightness, "blur_var": blur_var}

    # Fallback: whole image
    brightness = float(gray.mean() / 255.0)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    blur_var = float(lap.var())
    return {"brightness": brightness, "blur_var": blur_var}


 # -----------------------------
 # Mask utilities (OpenCV)
 # -----------------------------
def _refine_mask(mask_255: np.ndarray) -> np.ndarray:
    """Morphology cleanup + fill internal holes + keep up to 3 largest components."""
    m = (mask_255 > 0).astype(np.uint8) * 255

    # Smooth edges / remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)

    # Fill holes so dark/yellow/brown symptoms inside the leaf are NOT removed from the mask
    # (important for previews and any mask-based metrics)
    h, w = m.shape[:2]
    seed = None
    for sx, sy in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if m[sy, sx] == 0:
            seed = (sx, sy)
            break

    if seed is not None:
        flood = m.copy()
        ffmask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, ffmask, seedPoint=seed, newVal=255)
        holes = cv2.bitwise_not(flood)
        m = cv2.bitwise_or(m, holes)

    # Keep up to 3 biggest foreground components
    m_bin = (m > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m_bin, connectivity=8)
    if num <= 1:
        return (m_bin * 255).astype(np.uint8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx_sorted = np.argsort(areas)[::-1]
    keep = np.zeros_like(m_bin)
    for k in idx_sorted[:3]:
        keep[labels == (k + 1)] = 1

    return (keep * 255).astype(np.uint8)


def _segment_grabcut(np_rgb: np.ndarray, iters: int = 5) -> np.ndarray:
    """GrabCut foreground mask (0/255) initialized with a center rectangle."""
    bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    mask = np.zeros((h, w), np.uint8)
    pad_w = int(0.08 * w)
    pad_h = int(0.08 * h)
    rect = (pad_w, pad_h, w - 2 * pad_w, h - 2 * pad_h)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    return _refine_mask((fg * 255).astype(np.uint8))


def _segment_hsv_green(np_rgb: np.ndarray) -> np.ndarray:
    """HSV green-ish threshold mask (0/255)."""
    hsv = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2HSV)
    lower = np.array([25, 25, 25], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return _refine_mask(mask)


def _score_mask(mask_255: np.ndarray) -> float:
    """Heuristic score to auto-pick a leaf-like mask."""
    h, w = mask_255.shape[:2]
    img_area = float(h * w)
    cov = float((mask_255 > 0).mean())
    if cov < 0.03 or cov > 0.90:
        return -1.0

    cnts, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return -1.0

    areas = np.array([cv2.contourArea(c) for c in cnts], dtype=np.float32)
    largest = float(areas.max())
    largest_ratio = largest / img_area
    frag_penalty = float(len(cnts)) * 0.02
    return (0.6 * cov + 1.2 * largest_ratio) - frag_penalty


def _auto_leaf_mask(np_rgb: np.ndarray) -> tuple[np.ndarray, str]:
    """Try GrabCut + HSV and pick the best."""
    candidates = []
    try:
        m1 = _segment_grabcut(np_rgb, iters=5)
        candidates.append((m1, "grabcut"))
    except Exception:
        pass

    try:
        m2 = _segment_hsv_green(np_rgb)
        candidates.append((m2, "hsv"))
    except Exception:
        pass

    if not candidates:
        h, w = np_rgb.shape[:2]
        return np.zeros((h, w), dtype=np.uint8), "none"

    best_mask, best_name = max(candidates, key=lambda t: _score_mask(t[0]))
    return best_mask, best_name


def _bbox_from_mask_255(mask_255: np.ndarray, np_rgb: np.ndarray | None = None):
    """
    Choose a *best* component bbox, not simply the largest.

    Why: In real photos, the largest green region can be background leaves or even fruit.
    We score candidates using:
      - closeness to image center (users usually center the target leaf)
      - focus/sharpness (foreground leaf tends to be sharper than background)
      - leaf-like shape (less circular than fruit)
      - reasonable area
    """
    cnts, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    h, w = mask_255.shape[:2]
    img_area = float(h * w)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    # Precompute focus map if image is provided
    gray = None
    if np_rgb is not None and np_rgb.ndim == 3:
        try:
            gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
        except Exception:
            gray = None

    best_bbox = None
    best_score = -1e9

    for c in cnts:
        area = float(cv2.contourArea(c))
        if area <= 0:
            continue

        # Ignore extremely tiny components
        if (area / img_area) < 0.005:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        x1 = x + bw - 1
        y1 = y + bh - 1

        # Distance to image center (normalized)
        bx, by = x + bw / 2.0, y + bh / 2.0
        dist = float(np.hypot((bx - cx) / w, (by - cy) / h))

        # Shape: circularity (fruit ~1, leaves usually lower)
        perim = float(cv2.arcLength(c, True))
        circularity = (4.0 * np.pi * area) / (perim * perim + 1e-6)
        circularity = float(np.clip(circularity, 0.0, 1.2))

        # Focus (foreground tends to be sharper); normalize to ~[0,1.5]
        focus_norm = 0.0
        if gray is not None:
            roi = gray[max(0, y):min(h, y1 + 1), max(0, x):min(w, x1 + 1)]
            if roi.size >= 400:  # avoid unstable tiny ROIs
                fv = float(cv2.Laplacian(roi, cv2.CV_64F).var())
                focus_norm = float(np.clip(fv / 120.0, 0.0, 1.5))

        # Border penalty (background regions often touch borders)
        touches_border = (x <= 1) or (y <= 1) or (x1 >= w - 2) or (y1 >= h - 2)
        border_penalty = 0.35 if touches_border else 0.0

        area_ratio = area / img_area

        # Candidate score (tuned for "choose the centered, sharp leaf")
        score = (
            0.80 * area_ratio
            - 1.60 * dist
            + 0.70 * focus_norm
            + 0.35 * (1.0 - min(circularity, 1.0))
            - border_penalty
        )

        if score > best_score:
            best_score = score
            best_bbox = (int(x), int(y), int(x1), int(y1))

    # Fallback: if scoring filtered everything, use largest area
    if best_bbox is None:
        c = max(cnts, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        best_bbox = (int(x), int(y), int(x + bw - 1), int(y + bh - 1))

    return best_bbox


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


def _overlay_mask_on_crop(crop_img: Image.Image, mask_crop_255: np.ndarray) -> Image.Image:
    """
    Preview-only: draw mask OUTLINE on the ORIGINAL crop.

    Important: Do NOT fill/tint the whole leaf region, otherwise disease spots can look
    "greener" or visually muted. Outlines preserve the original colors and symptoms.
    """
    arr = np.asarray(crop_img.convert("RGB"), dtype=np.uint8)
    if mask_crop_255 is None or not bool(np.any(mask_crop_255 > 0)):
        return crop_img

    cnts, _ = cv2.findContours(mask_crop_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return crop_img

    out = arr.copy()
    # Drawing on RGB works fine because (0,255,0) is green in both RGB and BGR.
    cv2.drawContours(out, cnts, -1, (0, 255, 0), thickness=3)
    return Image.fromarray(out)


def mask_leaf_for_prediction(img: Image.Image):
    """
    Returns:
      pred_img  -> ORIGINAL crop (keeps symptoms!)
      preview_img -> masked preview (optional UI)
      info
    """
    W, H = img.size

    np_rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)

    # Auto-pick best mask between GrabCut and HSV
    mask_255, method = _auto_leaf_mask(np_rgb)
    kept_ratio = float((mask_255 > 0).mean())

    if kept_ratio >= KEPT_RATIO_MIN:
        # Use the image itself to help select the most relevant component (sharp + centered)
        bbox = _bbox_from_mask_255(mask_255, np_rgb=np_rgb)
        if bbox is not None:
            bbox = _pad_bbox(bbox, W, H)
            x0, y0, x1, y1 = bbox

            # ORIGINAL crop used for prediction ✅ (symptoms preserved)
            pred_img = img.crop((x0, y0, x1 + 1, y1 + 1))

            # Mask crop for quality metrics + preview overlay
            mask_crop_255 = mask_255[y0:y1 + 1, x0:x1 + 1]

            # Preview: overlay mask on the ORIGINAL crop (no symptom removal)
            preview = _overlay_mask_on_crop(pred_img, mask_crop_255)

            info = {"kept_ratio": kept_ratio, "method": method, "bbox": bbox, "mask_crop_255": mask_crop_255}
            return pred_img, preview, info

    # 3) final fallback: no masking, no block
    return img, img, {"kept_ratio": 1.0, "method": "none", "bbox": None, "mask_crop_255": None}


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

**How the app works now:**
- The app tries to **find the leaf area** and crops the image.
- The model predicts on the **original cropped leaf** (symptoms are preserved).
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

    # -----------------------------
    # NEW behavior: find bbox, then predict on ORIGINAL cropped leaf
    # -----------------------------
    pred_img, preview_img, mask_info = mask_leaf_for_prediction(img)

    with st.expander("🧪 Show leaf crop / masking preview (used to locate the leaf)", expanded=False):
        st.image(preview_img, use_container_width=True)
        st.caption(f"Method: {mask_info['method']} | Kept ratio: {mask_info['kept_ratio']:.2%}")
        if mask_info["method"] == "none":
            st.info("Masking wasn’t reliable here — using the original image for prediction (no crop).")

    # Quality checks on the image we actually feed to the model
    q = image_quality(pred_img, mask_info.get("mask_crop_255"))
    if q["brightness"] < BRIGHTNESS_MIN or q["blur_var"] < BLUR_VAR_MIN:
        st.warning("⚠️ The image is blur or low quality, please upload another photo and try again.")
        st.stop()

    # -----------------------------
    # Predict
    # -----------------------------
    x = preprocess(pred_img, model)

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

    st.caption("Tip: If predictions look wrong, try a brighter/sharper photo.")
