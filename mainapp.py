hqckz
hqckz
Invisible

a pred — 17:59
mitsu — 19:02
ashveen im so hungry i could eat you rn and never change a shade
hqckz
 — 19:03
What does that even mean
Randomer
 — 19:03
mathew so hungry he ate himself to sleep
hqckz
 — 19:05
Mathew so hungry he ate himself
lurker

 — 19:06
Hungry so mathew he ate him
Randomer
 — 19:26
harvey is him
a pred — 19:29
wait its a spectre
ok get back to the truck we got the ghost
Randomer
 — 19:30
hqckz
 — 19:36
??
Ur watching??
chUd
Randomer
 — 19:36
no im saying ashveen favourite words
lets hop on valorant
little kids
hqckz
 — 19:37
Randomer
 — 19:46
im getting the israel backed fyp all ashveen fault
Randomer
 started a call. — 20:09
a pred — 21:15
GIF - IMG WA.jpg?ex=b&is=aff&hm=afebadaccabccdfaeeafbcccee&
Bk kinda fire
hqckz
 — 21:32
a GREED
Randomer
 — 21:33
https://67movies.net/
67movies.net
67movies.net
Stream thousands of movies and TV shows for free on 67movies.net. Enjoy trending titles and discover new favorites.
67 movies
Ashveen
 — 21:48
whatw
what did u even say
@a pred
other than wooden spoon
a pred — 21:56
Mcflurry taste buns nwo
hqckz
 — 21:58
what changed
hqckz
 — 22:22
Attachment file type: unknown
efficientnet_finetuned.h5
32.50 MB
Randomer
 — 22:34
we do not need to see allat mid
GIF - image.png?ex=bda&is=ca&hm=fcdfffdadbccfccfefbcdbdabdd&
hqckz
 — 23:12
# ----------------------------------------
# Force CPU mode to avoid silent GPU crash
# ----------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

message.txt
6 KB
Randomer
 — 23:24
https://www.instagram.com/reel/DU4QUAPgSDd/?igsh=aXNwaW1tYWtjcDlq

dylan.aitools
Choose your fighter #fyp #transition #clashroyale #clash
Likes
4753
Image

Instagram
﻿
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
