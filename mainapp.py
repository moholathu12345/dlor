# ----------------------------------------
# Force CPU mode to avoid silent GPU crash
# ----------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

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
        interpreter = tf.lite.Interpreter(model_path=r"sufiiswatchingme.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"❌ Failed to load TFLite model: {e}")
        st.stop()

interpreter = load_tflite_model()
st.success("TFLite model loaded successfully")

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------------------------------
# Image preprocessing
# ----------------------------------------
def preprocess_image(image, target_size=(28, 28)):
    # Convert to grayscale
    image = image.convert("L")

    # Resize to model input size
    image = image.resize(target_size)

    # Convert to numpy
    image = np.array(image).astype("float32") / 255.0

    # Add channel dimension if required
    image = np.expand_dims(image, axis=-1)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image

# ----------------------------------------
# Upload image
# ----------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_data = preprocess_image(image)

    # ----------------------------------------
    # TFLite Inference
    # ----------------------------------------
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    predicted_class = int(np.argmax(output_data))
    confidence = float(np.max(output_data))

    st.subheader("Prediction Result")
    st.write("Predicted class index:", predicted_class)
    st.write("Confidence:", f"{confidence:.2f}")
