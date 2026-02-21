# ----------------------------------------
# Planter: One-page Streamlit upload + results UI
# TensorFlow Lite inference + clean, modern layout
# ----------------------------------------

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime

from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Planter", layout="wide")


# -----------------------------
# Labels (order must match model output)
# -----------------------------
CLASS_NAMES = [
  "Apple___Apple_scab",
  "Apple___Black_rot",
  "Apple___Cedar_apple_rust",
  "Apple___healthy",
  "Blueberry___healthy",
  "Cherry_(including_sour)___Powdery_mildew",
  "Cherry_(including_sour)___healthy",
  "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
  "Corn_(maize)___Common_rust_",
  "Corn_(maize)___Northern_Leaf_Blight",
  "Corn_(maize)___healthy",
  "Grape___Black_rot",
  "Grape___Esca_(Black_Measles)",
  "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
  "Grape___healthy",
  "Orange___Haunglongbing_(Citrus_greening)",
  "Peach___Bacterial_spot",
  "Peach___healthy",
  "Pepper,_bell___Bacterial_spot",
  "Pepper,_bell___healthy",
  "Potato___Early_blight",
  "Potato___Late_blight",
  "Potato___healthy",
  "Raspberry___healthy",
  "Soybean___healthy",
  "Squash___Powdery_mildew",
  "Strawberry___Leaf_scorch",
  "Strawberry___healthy",
  "Tomato___Bacterial_spot",
  "Tomato___Early_blight",
  "Tomato___Late_blight",
  "Tomato___Leaf_Mold",
  "Tomato___Septoria_leaf_spot",
  "Tomato___Spider_mites Two-spotted_spider_mite",
  "Tomato___Target_Spot",
  "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
  "Tomato___Tomato_mosaic_virus",
  "Tomato___healthy"
]


def format_label(raw: str) -> str:
    """
    Convert: Tomato___Early_blight -> Tomato  |  Early blight
    Also cleans underscores and a few punctuation bits for nicer UI.
    """
    if "___" in raw:
        plant, condition = raw.split("___", 1)
    else:
        plant, condition = raw, "Unknown"

    plant = plant.replace("_", " ").replace("(maize)", "maize").strip()
    condition = condition.replace("_", " ").strip()

    # Slight readability tweaks
    condition = condition.replace("  ", " ")
    plant = plant.replace("  ", " ")

    return f"{plant} | {condition}"


# -----------------------------
# Cache the TFLite interpreter
# -----------------------------
@st.cache_resource
def load_interpreter(model_path: str):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


# -----------------------------
# Preprocess for TFLite (EfficientNet style for float models)
# -----------------------------
def preprocess_for_tflite(pil_img: Image.Image, input_details):
    in_shape = input_details[0]["shape"]
    in_dtype = input_details[0]["dtype"]

    # Shape: [1, H, W, C]
    H = int(in_shape[1])
    W = int(in_shape[2])
    C = int(in_shape[3])

    if C == 3:
        img = pil_img.convert("RGB")
    else:
        img = pil_img.convert("L")

    img = img.resize((W, H))
    x = np.array(img)

    if C == 1 and x.ndim == 2:
        x = np.expand_dims(x, axis=-1)

    x = np.expand_dims(x, axis=0)  # (1, H, W, C)

    if in_dtype == np.float32:
        x = x.astype(np.float32)
        x = eff_pre(x)  # matches EfficientNet training preprocess
    else:
        x = x.astype(in_dtype)

    return x


def softmax_if_needed(vec: np.ndarray) -> np.ndarray:
    """
    Some models output logits, some output probabilities.
    If values don't look like probabilities, softmax them.
    """
    v = vec.astype(np.float32)
    if np.any(v < 0) or (np.max(v) > 1.0) or (abs(np.sum(v) - 1.0) > 0.05):
        e = np.exp(v - np.max(v))
        return e / (np.sum(e) + 1e-9)
    return v


# -----------------------------
# UI Styling (inspired by your HTML page)
# -----------------------------
st.markdown(
    """
<style>
/* page background */
.stApp {
  background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
}

/* remove default header space */
header[data-testid="stHeader"] { background: transparent; }
div.block-container { padding-top: 1.2rem; }

/* top header bar */
.planter-topbar {
  position: sticky;
  top: 0;
  z-index: 50;
  padding: 1rem 0.8rem;
  border-radius: 14px;
  background: linear-gradient(135deg, #166534, #22c55e);
  box-shadow: 0 4px 6px -1px rgba(0,0,0,0.12);
  color: white;
  margin-bottom: 1.25rem;
}
.planter-brand {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-weight: 800;
  font-size: 1.35rem;
}
.planter-icon {
  width: 36px; height: 36px;
  border-radius: 10px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: rgba(255,255,255,0.22);
}

/* hero */
.planter-hero h1 {
  margin: 0.35rem 0 0.4rem 0;
  font-size: clamp(1.6rem, 3vw, 2.2rem);
  font-weight: 800;
  color: #0f172a;
}
.planter-hero p {
  margin: 0;
  color: #475569;
  font-size: 1.05rem;
}

/* card */
.planter-card {
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  padding: 1.25rem 1.25rem;
  box-shadow: 0 4px 6px -1px rgba(0,0,0,0.10);
}
.planter-card:hover {
  box-shadow: 0 10px 15px -3px rgba(0,0,0,0.10);
}
.planter-card-title {
  font-size: 1.1rem;
  font-weight: 700;
  color: #166534;
  padding-bottom: 0.65rem;
  margin-bottom: 0.9rem;
  border-bottom: 2px solid #dcfce7;
}

/* result highlight */
.planter-result {
  border-left: 4px solid #22c55e;
  background: linear-gradient(135deg, #ffffff 0%, #ecfdf5 100%);
}

/* footer */
.planter-footer {
  text-align: center;
  color: #64748b;
  font-size: 0.9rem;
  padding: 1.5rem 0 0.5rem 0;
  border-top: 1px solid #e2e8f0;
  margin-top: 1.25rem;
}
.small-muted { color: #64748b; font-size: 0.9rem; }

/* make Streamlit buttons look more premium */
.stButton > button {
  border-radius: 10px !important;
  padding: 0.65rem 1rem !important;
  font-weight: 700 !important;
  border: 0 !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #166534, #14532d) !important;
  color: white !important;
  box-shadow: 0 4px 6px -1px rgba(0,0,0,0.12) !important;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-1px);
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Top bar + hero
# -----------------------------
st.markdown(
    """
<div class="planter-topbar">
  <div class="planter-brand">
    <div class="planter-icon">🌱</div>
    <div>Planter</div>
  </div>
</div>
<div class="planter-hero">
  <h1>Plant disease detection</h1>
  <p>Upload a leaf photo to predict the disease class and confidence score.</p>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Session state: history
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # each: {label, conf, time}

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = "sufiiswatchingme.tflite"

try:
    interpreter = load_interpreter(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model file: {e}")
    st.stop()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# -----------------------------
# Layout: main (upload/results) + sidebar (history)
# -----------------------------
left, right = st.columns([1.8, 1.0], gap="large")

with left:
    st.markdown('<div class="planter-card">', unsafe_allow_html=True)
    st.markdown('<div class="planter-card-title">Upload leaf image</div>', unsafe_allow_html=True)

    with st.expander("Image guidelines", expanded=False):
        st.write(
            "- Use a single leaf\n"
            "- Good lighting\n"
            "- Avoid blur\n"
            "- Leaf should fill most of the frame"
        )

    uploaded_file = st.file_uploader(
        "Choose an image (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )

    img = None
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
        except Exception:
            st.error("That file could not be read as an image. Try a different JPG/PNG.")
            img = None

    if img is not None:
        st.image(img, caption="Preview", use_container_width=True)

    colA, colB = st.columns([1, 1])
    with colA:
        analyze = st.button("Analyze image", type="primary", disabled=(img is None))
    with colB:
        clear = st.button("Clear", disabled=(img is None))

    if clear:
        st.session_state.last_result = None
        st.rerun()

    st.markdown('<p class="small-muted">Tip: A clear, well-lit photo of a single leaf gives the best results.</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Results
    if analyze and img is not None:
        with st.spinner("Analyzing image with AI model..."):
            try:
                input_data = preprocess_for_tflite(img, input_details)

                # Safety guard to prevent silent shape mismatches
                expected = tuple(input_details[0]["shape"])
                got = tuple(input_data.shape)
                if got != expected:
                    st.error(f"Input tensor mismatch. Expected {expected}, got {got}.")
                    st.stop()

                interpreter.set_tensor(input_details[0]["index"], input_data)
                interpreter.invoke()

                output = interpreter.get_tensor(output_details[0]["index"])
                vec = output[0]
                probs = softmax_if_needed(vec)

                pred_idx = int(np.argmax(probs))
                conf = float(np.max(probs))

                # Top 3
                top_k = 3
                top_idx = probs.argsort()[-top_k:][::-1]
                top_items = []
                for i in top_idx:
                    name = CLASS_NAMES[int(i)] if int(i) < len(CLASS_NAMES) else f"class_{int(i)}"
                    top_items.append((format_label(name), float(probs[int(i)])))

                raw_label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"class_{pred_idx}"
                pretty = format_label(raw_label)

                st.session_state.last_result = {
                    "label": pretty,
                    "confidence": conf,
                    "top": top_items,
                }

                # Push into history
                st.session_state.history.insert(0, {
                    "label": pretty,
                    "confidence": conf,
                    "time": datetime.now().strftime("%H:%M"),
                })
                st.session_state.history = st.session_state.history[:10]

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    if st.session_state.last_result is not None:
        res = st.session_state.last_result
        confidence_pct = int(round(res["confidence"] * 100))

        st.markdown('<div class="planter-card planter-result">', unsafe_allow_html=True)
        st.markdown('<div class="planter-card-title">Analysis complete</div>', unsafe_allow_html=True)

        st.markdown(
            f"<div style='text-align:center; font-size:1.35rem; font-weight:800; color:#166534; margin:0.25rem 0 0.6rem 0;'>{res['label']}</div>",
            unsafe_allow_html=True
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            st.write("Confidence")
        with c2:
            st.write(f"**{confidence_pct}%**")

        st.progress(res["confidence"])

        if confidence_pct < 60:
            st.warning("Low confidence. Retake the photo with better lighting and less blur.")

        st.markdown("Top predictions")
        for name, p in res["top"]:
            st.write(f"- {name} ({p*100:.1f}%)")

        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="planter-card">', unsafe_allow_html=True)
    st.markdown('<div class="planter-card-title">Recent analyses</div>', unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.write("No analyses yet. Upload an image to get started.")
    else:
        for item in st.session_state.history:
            conf_pct = int(round(item["confidence"] * 100))
            st.markdown(
                f"""
<div style="padding:0.75rem; border:1px solid #e2e8f0; border-radius:12px; background:#f8fafc; margin-bottom:0.6rem;">
  <div style="font-weight:700; color:#0f172a; font-size:0.95rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
    {item["label"]}
  </div>
  <div style="display:flex; justify-content:space-between; color:#64748b; font-size:0.85rem; margin-top:0.25rem;">
    <span>{item["time"]}</span>
    <span style="font-weight:800; color:#166534;">{conf_pct}%</span>
  </div>
</div>
""",
                unsafe_allow_html=True
            )

    if st.button("Clear history"):
        st.session_state.history = []
        st.session_state.last_result = None
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    """
<div class="planter-footer">
  <div>Planter v1.0 • AI-powered plant health monitoring</div>
  <div style="margin-top:0.25rem; font-size:0.82rem;">
    For educational and research use. Not a substitute for professional agronomic advice.
  </div>
</div>
""",
    unsafe_allow_html=True
)
