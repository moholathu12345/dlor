# ----------------------------------------
# Force CPU mode to avoid silent GPU crash
# ----------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import base64
from datetime import datetime

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# EfficientNet preprocessing (only used when model input is float32)
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre


# ----------------------------------------
# Classes
# ----------------------------------------
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


# ----------------------------------------
# Page configuration
# ----------------------------------------
st.set_page_config(page_title="Planter - Plant Rotting Detection", page_icon="🌱", layout="wide")


# ----------------------------------------
# Styles (ported from your HTML)
# ----------------------------------------
CSS = """
<style>
  :root {
    --bg-primary: #ffffff;
    --bg-secondary: #f8fafc;
    --bg-tertiary: #ecfdf5;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #64748b;
    --border-color: #e2e8f0;
    --accent-primary: #166534;
    --accent-primary-hover: #14532d;
    --accent-secondary: #22c55e;
    --accent-light: #dcfce7;
    --warning: #f59e0b;
    --warning-bg: #fffbeb;
    --error: #ef4444;
    --error-bg: #fef2f2;
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
    --radius-md: 0.5rem;
    --radius-lg: 0.75rem;
    --radius-xl: 1rem;
    --transition-fast: 150ms ease;
    --container-max: 1100px;
    --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }

  .stApp {
    font-family: var(--font-family);
    background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
    color: var(--text-primary);
  }

  .planter-header {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    color: white;
    padding: 1.1rem 0;
    border-radius: var(--radius-xl);
    box-shadow: var(--shadow-md);
    margin-bottom: 1.25rem;
  }
  .planter-header-inner {
    max-width: var(--container-max);
    margin: 0 auto;
    padding: 0 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
  }
  .logo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-weight: 700;
    font-size: 1.35rem;
  }
  .logo-icon {
    width: 36px;
    height: 36px;
    background: rgba(255,255,255,0.2);
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.15rem;
  }

  .hero {
    max-width: var(--container-max);
    margin: 0 auto;
    padding: 0 1.25rem;
    text-align: center;
    margin-bottom: 1.25rem;
  }
  .hero-title {
    font-size: clamp(1.75rem, 4vw, 2.25rem);
    font-weight: 700;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, var(--accent-primary), var(--text-primary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-secondary);
    max-width: 680px;
    margin: 0 auto;
  }

  .card {
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-xl);
    padding: 1.25rem;
    box-shadow: var(--shadow-md);
    transition: box-shadow var(--transition-fast);
  }
  .card:hover { box-shadow: var(--shadow-lg); }

  .card-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--accent-primary);
    padding-bottom: 0.75rem;
    margin-bottom: 1rem;
    border-bottom: 2px solid var(--accent-light);
  }

  .result-card {
    border-left: 4px solid var(--accent-secondary);
    background: linear-gradient(135deg, #ffffff 0%, var(--bg-tertiary) 100%);
  }

  .result-label {
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--accent-primary);
    margin: 0.4rem 0 0.6rem 0;
    text-align: center;
  }

  .confidence-wrap { margin: 0.5rem 0 0.25rem 0; }
  .confidence-header {
    display: flex;
    justify-content: space-between;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 0.35rem;
  }
  .confidence-bar {
    height: 10px;
    background: var(--border-color);
    border-radius: var(--radius-lg);
    overflow: hidden;
  }
  .confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent-secondary), var(--accent-primary));
    border-radius: var(--radius-lg);
    width: 0%;
  }

  .warning-box {
    background: var(--warning-bg);
    border-left: 4px solid var(--warning);
    padding: 0.75rem;
    border-radius: 0 var(--radius-md) var(--radius-md) 0;
    margin-top: 0.9rem;
    color: #92400e;
    font-weight: 600;
    font-size: 0.95rem;
  }

  .muted { color: var(--text-muted); font-size: 0.9rem; }

  div.stButton > button {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-primary-hover));
    color: white;
    border: 0;
    padding: 0.75rem 1.2rem;
    border-radius: var(--radius-md);
    font-weight: 700;
    min-height: 44px;
    box-shadow: var(--shadow-md);
    transition: transform var(--transition-fast);
  }
  div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-lg);
  }

  .history-item {
    display: flex;
    gap: 0.75rem;
    padding: 0.75rem;
    background: var(--bg-secondary);
    border-radius: var(--radius-md);
    border: 1px solid var(--border-color);
    margin-bottom: 0.6rem;
  }
  .history-thumb {
    width: 56px;
    height: 56px;
    border-radius: 0.375rem;
    object-fit: cover;
    border: 1px solid var(--border-color);
    flex-shrink: 0;
  }
  .history-label {
    font-weight: 800;
    color: var(--text-primary);
    font-size: 0.95rem;
    margin-bottom: 0.1rem;
  }
  .history-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: var(--text-muted);
  }
  .history-confidence {
    font-weight: 800;
    color: var(--accent-primary);
  }

  .footer {
    max-width: var(--container-max);
    margin: 1.75rem auto 0 auto;
    padding: 1.25rem;
    text-align: center;
    border-top: 1px solid var(--border-color);
    color: var(--text-muted);
    font-size: 0.9rem;
  }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------
# Header + Hero
# ----------------------------------------
st.markdown(
    """
    <div class="planter-header" role="banner">
      <div class="planter-header-inner">
        <div class="logo">
          <div class="logo-icon">🌱</div>
          <div>Planter</div>
        </div>
        <div style="font-weight:600; opacity:0.95;">History</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-title">Detect Plant Rotting with AI</div>
      <div class="hero-subtitle">
        Upload a leaf photo to instantly analyze for disease, rot, or decay.
        Get confidence scores and actionable insights.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------
# TFLite model loading (cached)
# ----------------------------------------
@st.cache_resource
def load_tflite_model(model_path: str):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

MODEL_PATH = "sufiiswatchingme.tflite"

try:
    interpreter = load_tflite_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load TFLite model: {e}")
    st.stop()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

in_shape = input_details[0]["shape"]   # [1,H,W,C]
in_dtype = input_details[0]["dtype"]   # np.float32 or np.uint8, etc.

H = int(in_shape[1])
W = int(in_shape[2])
C = int(in_shape[3])


# ----------------------------------------
# Helpers
# ----------------------------------------
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    # Channels
    if C == 3:
        img = pil_img.convert("RGB")
    else:
        img = pil_img.convert("L")

    # Resize
    img = img.resize((W, H))

    # To numpy
    x = np.array(img)

    # If grayscale, ensure channel dimension exists
    if C == 1 and x.ndim == 2:
        x = np.expand_dims(x, axis=-1)

    # Batch dimension
    x = np.expand_dims(x, axis=0)

    # Dtype handling
    if in_dtype == np.float32:
        x = x.astype(np.float32)
        x = eff_pre(x)  # matches your training preprocessing
    else:
        x = x.astype(in_dtype)

    return x

def predict_tflite(pil_img: Image.Image):
    x = preprocess_image(pil_img)

    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    probs = output[0]
    pred_idx = int(np.argmax(probs))
    conf = float(np.max(probs))

    label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"class_{pred_idx}"
    return label, conf, probs

def image_bytes_to_data_uri(img_bytes: bytes) -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ----------------------------------------
# Session state for history
# ----------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # newest first, each {img_bytes,label,conf,ts}


# ----------------------------------------
# Main layout: left (upload/results) + right (history)
# ----------------------------------------
left, right = st.columns([1.25, 0.85], gap="large")

with left:
    st.markdown('<div class="card"><div class="card-title">📤 Upload Leaf Image</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload leaf image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    st.markdown('<div class="muted">💡 Tip: Use a clear, well-lit photo of a single leaf for best results.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    pil_image = None
    img_bytes = None

    if uploaded is not None:
        pil_image = Image.open(uploaded)
        img_bytes = uploaded.getvalue()

        st.markdown('<div class="card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.image(pil_image, caption=f"{uploaded.name} • {uploaded.size/1024:.1f} KB", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    analyze_disabled = pil_image is None
    if st.button("🔍 Analyze Image", disabled=analyze_disabled):
        with st.spinner("Analyzing image with AI model..."):
            label, conf, probs = predict_tflite(pil_image)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")

            st.session_state.history.insert(
                0,
                {"img_bytes": img_bytes, "label": label, "conf": conf, "ts": ts}
            )
            st.session_state.history = st.session_state.history[:10]

            # Store latest probs (optional)
            st.session_state.latest_probs = probs

    # Results (show latest if exists)
    if st.session_state.history:
        latest = st.session_state.history[0]
        conf_pct = int(round(latest["conf"] * 100))

        st.markdown('<div class="card result-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">✅ Analysis Complete</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-label">{latest["label"]}</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="confidence-wrap">
              <div class="confidence-header">
                <span>Confidence</span>
                <span>{conf_pct}%</span>
              </div>
              <div class="confidence-bar" role="progressbar" aria-valuenow="{conf_pct}" aria-valuemin="0" aria-valuemax="100">
                <div class="confidence-fill" style="width:{conf_pct}%;"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if conf_pct < 70:
            st.markdown(
                '<div class="warning-box">⚠️ Low confidence. Retake photo with better lighting for more accurate results.</div>',
                unsafe_allow_html=True,
            )

        # Optional raw output
        with st.expander("Show raw model output"):
            probs = getattr(st.session_state, "latest_probs", None)
            if probs is None:
                st.write("No output available yet.")
            else:
                st.write(probs)

        # Optional top 5
        with st.expander("Show top 5 predictions"):
            probs = getattr(st.session_state, "latest_probs", None)
            if probs is None:
                st.write("No output available yet.")
            else:
                topk = np.argsort(probs)[::-1][:5]
                for i in topk:
                    name = CLASS_NAMES[int(i)] if int(i) < len(CLASS_NAMES) else f"class_{int(i)}"
                    st.write(f"{name}: {float(probs[int(i)])*100:.2f}%")

        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card"><div class="card-title">📋 Recent Analyses</div>', unsafe_allow_html=True)
    st.caption(f"Model input: shape={in_shape}, dtype={in_dtype}")

    if not st.session_state.history:
        st.markdown(
            '<div class="muted" style="text-align:center; padding: 1.2rem 0;">No analyses yet. Upload an image to get started.</div>',
            unsafe_allow_html=True,
        )
    else:
        for item in st.session_state.history:
            conf_pct = int(round(item["conf"] * 100))
            thumb_uri = image_bytes_to_data_uri(item["img_bytes"])

            st.markdown(
                f"""
                <div class="history-item">
                  <img class="history-thumb" src="{thumb_uri}" alt="history thumbnail">
                  <div style="flex:1; min-width:0;">
                    <div class="history-label">{item["label"]}</div>
                    <div class="history-meta">
                      <span>{item["ts"]}</span>
                      <span class="history-confidence">{conf_pct}%</span>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if st.button("Clear History", use_container_width=True):
        st.session_state.history = []
        if "latest_probs" in st.session_state:
            del st.session_state.latest_probs
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="footer" role="contentinfo">
      <div>Planter v1.0 • AI-Powered Plant Health Monitoring</div>
      <div style="margin-top:0.25rem; font-size:0.85rem;">
        For agricultural research use. Not a substitute for professional agronomic advice.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
