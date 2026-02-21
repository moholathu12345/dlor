# ----------------------------------------
# Force CPU mode (prevents GPU crash on cloud)
# ----------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import base64
from datetime import datetime

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre


# ----------------------------------------
# CLASS NAMES
# ----------------------------------------
CLASS_NAMES = [
  "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
  "Blueberry___healthy","Cherry_(including_sour)___Powdery_mildew","Cherry_(including_sour)___healthy",
  "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
  "Corn_(maize)___Northern_Leaf_Blight","Corn_(maize)___healthy",
  "Grape___Black_rot","Grape___Esca_(Black_Measles)",
  "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Grape___healthy",
  "Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy",
  "Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy",
  "Potato___Early_blight","Potato___Late_blight","Potato___healthy",
  "Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew",
  "Strawberry___Leaf_scorch","Strawberry___healthy",
  "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight",
  "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot",
  "Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
  "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus",
  "Tomato___healthy"
]


# ----------------------------------------
# PAGE CONFIG
# ----------------------------------------
st.set_page_config(
    page_title="Planter - Plant Rotting Detection",
    page_icon="🌱",
    layout="wide"
)


# ----------------------------------------
# LOAD TFLITE MODEL
# ----------------------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="sufiiswatchingme.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_model()
except Exception as e:
    st.error(f"Failed to load TFLite model: {e}")
    st.stop()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

in_shape = input_details[0]["shape"]
in_dtype = input_details[0]["dtype"]

H = int(in_shape[1])
W = int(in_shape[2])
C = int(in_shape[3])


# ----------------------------------------
# PREPROCESSING
# ----------------------------------------
def preprocess_image(pil_img):
    if C == 3:
        img = pil_img.convert("RGB")
    else:
        img = pil_img.convert("L")

    img = img.resize((W, H))
    x = np.array(img)

    if C == 1 and x.ndim == 2:
        x = np.expand_dims(x, axis=-1)

    x = np.expand_dims(x, axis=0)

    if in_dtype == np.float32:
        x = x.astype(np.float32)
        x = eff_pre(x)
    else:
        x = x.astype(in_dtype)

    return x


def predict_image(pil_img):
    x = preprocess_image(pil_img)
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    probs = output[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))

    label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"class_{pred_idx}"
    return label, confidence, probs


# ----------------------------------------
# SESSION STATE
# ----------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# ----------------------------------------
# HEADER
# ----------------------------------------
st.markdown(
    """
    <div style="background:linear-gradient(135deg,#166534,#22c55e);
                padding:1rem;border-radius:1rem;margin-bottom:1rem;">
        <h2 style="color:white;margin:0;">🌱 Planter</h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("### Detect Plant Rotting with AI")
st.write("Upload a leaf image to analyze disease and get confidence scores.")


# ----------------------------------------
# LAYOUT
# ----------------------------------------
left, right = st.columns([1.3, 0.7])


# ----------------------------------------
# LEFT SIDE (UPLOAD + RESULTS)
# ----------------------------------------
with left:
    uploaded = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

    pil_image = None
    img_bytes = None

    if uploaded is not None:
        pil_image = Image.open(uploaded)
        img_bytes = uploaded.getvalue()

        st.image(pil_image,
                 caption=f"{uploaded.name} • {uploaded.size/1024:.1f} KB",
                 use_column_width=True)

    if st.button("🔍 Analyze Image", disabled=(pil_image is None)):
        with st.spinner("Analyzing image..."):
            label, conf, probs = predict_image(pil_image)

            st.session_state.history.insert(
                0,
                {
                    "img": img_bytes,
                    "label": label,
                    "conf": conf,
                    "time": datetime.now().strftime("%H:%M")
                }
            )
            st.session_state.history = st.session_state.history[:10]

            st.success("Analysis Complete")

            conf_pct = int(conf * 100)
            st.markdown(f"### {label}")
            st.progress(conf)
            st.write(f"Confidence: **{conf_pct}%**")

            if conf_pct < 70:
                st.warning("Low confidence. Try retaking the image with better lighting.")

            with st.expander("Top 5 Predictions"):
                top5 = np.argsort(probs)[::-1][:5]
                for i in top5:
                    name = CLASS_NAMES[int(i)]
                    st.write(f"{name}: {float(probs[int(i)])*100:.2f}%")


# ----------------------------------------
# RIGHT SIDE (HISTORY)
# ----------------------------------------
with right:
    st.markdown("### 📋 Recent Analyses")

    if not st.session_state.history:
        st.write("No analyses yet.")
    else:
        for item in st.session_state.history:
            st.image(item["img"], width=80)
            st.write(f"**{item['label']}**")
            st.write(f"{int(item['conf']*100)}% • {item['time']}")
            st.markdown("---")

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()


# ----------------------------------------
# FOOTER
# ----------------------------------------
st.markdown(
    """
    ---
    Planter v1.0 • AI-Powered Plant Health Monitoring  
    For research use only.
    """
)
