# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Detection (CNN)")
st.write("Upload a plant leaf image to detect its disease.")

model = tf.keras.models.load_model("models/plant_disease_model.h5")

# Mapping class indices
class_names = list(model.class_names) if hasattr(model, "class_names") else [
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    # Add all your folder class names in the same order as the training generator
]

uploaded = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded:
    img = Image.open(uploaded).resize((128, 128))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred = model.predict(img_array)
    st.success(f"🔬 Predicted: **{class_names[np.argmax(pred)]}**")
