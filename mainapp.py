# ----------------------------------------
# Force CPU mode to avoid silent GPU crash
# ----------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Optional: only used if model input is float32 and expects EfficientNet preprocessing
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre

# ----------------------------------------
# Page configuration
# ----------------------------------------
st.set_page_config(
    page_title="TensorFlow Lite Model Deployment",
    layout="centered"
)

st.title("TensorFlow Lite Model Deployment with Streamlit")
st.write("Upload an image to get a prediction.")

# ----------------------------------------
# Load TFLite model (cached)
# ----------------------------------------
@st.cache_resource
def load_tflite_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="sufiiswatchingme.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Failed to load TFLite model: {e}")
        st.stop()

interpreter = load_tflite_model()
st.success("TFLite model loaded successfully")

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read expected input shape/dtype from the model
in_shape = input_details[0]["shape"]      # e.g. [1,224,224,3]
in_dtype = input_details[0]["dtype"]      # e.g. np.float32 or np.uint8

# Some models may have dynamic batch size; we only care about H,W,C
H = int(in_shape[1])
W = int(in_shape[2])
C = int(in_shape[3])

st.caption(f"Model input: shape={in_shape}, dtype={in_dtype}")

# ----------------------------------------
# Image preprocessing
# ----------------------------------------
def preprocess_image(pil_img):
    # Ensure correct channels (most EfficientNet models want RGB)
    if C == 3:
        img = pil_img.convert("RGB")
    else:
        # fallback if model expects 1 channel
        img = pil_img.convert("L")

    # Resize to model expected size
    img = img.resize((W, H))

    # To numpy
    x = np.array(img)

    # Ensure channel dimension exists for grayscale case
    if C == 1 and x.ndim == 2:
        x = np.expand_dims(x, axis=-1)

    # Add batch dimension: (1, H, W, C)
    x = np.expand_dims(x, axis=0)

    # Handle dtype expectations
    if in_dtype == np.float32:
        x = x.astype(np.float32)
        # EfficientNet preprocessing matches your training pipeline
        # (EfficientNet preprocess expects 0..255 float, then transforms)
        x = eff_pre(x)
    else:
        # Quantized models typically expect uint8 0..255
        x = x.astype(in_dtype)

    return x

# ----------------------------------------
# Upload image
# ----------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_data = preprocess_image(image)

    # ----------------------------------------
    # TFLite Inference
    # ----------------------------------------
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    predicted_class = int(np.argmax(output_data, axis=-1)[0])
    confidence = float(np.max(output_data, axis=-1)[0])

    st.subheader("Prediction Result")
    st.write("Predicted class index:", predicted_class)
    st.write("Confidence:", f"{confidence:.4f}")



    probs = output_data[0]
predicted_class = int(np.argmax(probs))
confidence = float(np.max(probs))

pred_label = CLASS_NAMES[predicted_class] if predicted_class < len(CLASS_NAMES) else f"class_{predicted_class}"

st.subheader("Prediction Result")
st.write("Prediction:", pred_label)
st.write("Confidence:", f"{confidence*100:.1f}%")



    # Optional: show raw logits/probabilities
    with st.expander("Show raw model output"):
        st.write(output_data)
