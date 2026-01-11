# 🌿 Plant Disease Classifier (TensorFlow + Streamlit)

This repository contains a complete **image classification** project for plant disease detection:
- **Model training & evaluation** (Jupyter notebook)
- Optional **script-based pipeline** (`modeling/`)
- A ready-to-run **Streamlit web app** (`app.py`) for inference (upload image → predict)
- Saved Keras model(s) in `models/` and label mapping in `class_names.json`

---

## ✅ What’s inside

- **Streamlit app**: `app.py`
- **Model file** (example): `models/image_classification_model_linux.keras`
- **Class labels**: `class_names.json`
- **Deployment dependencies**: `requirements_capstone.txt` (recommended for deployment)
- **Optional Streamlit Cloud** deps: `requirements.txt` (if you use it for Cloud)

---

## 📁 Project structure

```
.
├── app.py
├── class_names.json
├── requirements.txt                 # optional (Streamlit Cloud)
├── requirements_capstone.txt        # deployment-only dependencies (no notebook/dev libs)
├── .python-version
├── Makefile
├── models/
│   ├── image_classification_model_linux.keras
│   └── .gitkeep
├── notebooks/
│   └── image_clasification_cnn_model.ipynb
└── modeling/
    ├── config.py
    ├── feature_engineering.py
    ├── train.py
    └── predict.py
```

> Note: Only the **essential notebook** is kept in `notebooks/`. All unnecessary notebooks were removed to keep the repo clean.

---

## 🧩 Requirements

### Python
- Recommended: **Python 3.11.3** (or any compatible Python 3.11.x)

### Deployment dependencies
The `requirements_capstone.txt` file contains the libraries needed for **deployment** (model / dashboard / Streamlit),
so it **does not include Jupyter or development-only libraries**.

---

## ⚡ Setup (recommended)

### Option 1 — Use the Makefile (fast)
If your `Makefile` contains a setup target, run:

```bash
make setup
```

If `make setup` is not available on your system, use the manual setup below.

---

## 🛠️ Manual setup (Windows / macOS / Linux)

### ✅ Windows — PowerShell

```PowerShell
pyenv local 3.11.3
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_capstone.txt
```

### ✅ Windows — Git Bash

```BASH
pyenv local 3.11.3
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements_capstone.txt
```

### ✅ macOS / Linux (Bash or Zsh)

```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements_capstone.txt
```

---

## ▶️ Run the Streamlit app locally

From the project root (where `app.py` is located):

```bash
streamlit run app.py
```

If `streamlit` is not recognized, use:

```bash
python -m streamlit run app.py
```

---

## 🧠 Model + labels (very important)

### Model file
Put your model inside the `models/` folder, for example:

- `models/image_classification_model_linux.keras`

Your Streamlit sidebar **Model path** must match the filename exactly.

### Class labels
`class_names.json` must be a JSON list, for example:

```json
["Early_blight", "Late_blight", "healthy"]
```

✅ **Order matters:** the list order must match the model output order.

---

## 📓 Training notebook

Notebook location:

- `notebooks/image_clasification_cnn_model.ipynb`

Typical workflow:
1. Load dataset (e.g., PlantVillage)
2. Train CNN / transfer learning model
3. Save trained model into `models/`
4. Export class order into `class_names.json`

---

## ☁️ Deploy on Streamlit Community Cloud

### 1) Ensure these files are in GitHub
- `app.py`
- `requirements.txt`
- `class_names.json`
- `models/image_classification_model_linux.keras`

### 2) Streamlit Cloud settings
- **Repository**: your GitHub repo
- **Branch**: `main`
- **Main file path**: `app.py`

### 3) Requirements file note (important)
Streamlit Cloud automatically reads `requirements.txt`.

---

## 🧯 Troubleshooting

### “Model file not found”
- Confirm the model exists in GitHub under `models/`
- Confirm the Streamlit sidebar path matches the filename exactly
- Check `.gitignore` rules: do not accidentally ignore `models/*.keras`

### “Layer conv2d expected variables, got 0”
This may happen when a `.keras` file saved on Windows is loaded on Linux.
Fix: export a Linux-friendly `.keras` model (or use the provided `*_linux.keras`) and commit it to `models/`.

---

## 📜 License
MIT — see `LICENSE`.
