from pathlib import Path
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


MODEL_PATH = Path("models/deep_cnn_best.keras")
CLASSES_PATH = Path("models/deep_cnn_classes.json")
DEFAULT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
IMAGE_SIZE = 64


def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def load_class_names():
    if CLASSES_PATH.exists():
        with open(CLASSES_PATH, encoding="utf-8") as file:
            return json.load(file)
    return DEFAULT_CLASSES


def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    array = tf.keras.utils.img_to_array(image)
    return np.expand_dims(array, axis=0)


st.set_page_config(page_title="OrbitalWatch Vision", page_icon="OV", layout="centered")
st.title("OrbitalWatch Vision")
st.caption("Classificação de imagens satelitais — CNN treinada do zero.")

model = load_model()
class_names = load_class_names()

if model is None:
    st.warning(
        "Modelo não encontrado. Treine com: "
        "`python src/train.py --data-dir dataset_eurosat --model deep_cnn --epochs 25 --image-size 64`"
    )

uploaded_file = st.file_uploader("Envie uma imagem satelital", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem enviada", use_container_width=True)

    if model is not None:
        probabilities = model.predict(preprocess_image(image), verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_class = class_names[predicted_index]
        confidence = probabilities[predicted_index] * 100

        st.subheader(f"Classe detectada: {predicted_class}")
        st.metric("Confiança", f"{confidence:.2f}%")

        st.write("Probabilidades por classe")
        for class_name, probability in zip(class_names, probabilities):
            st.progress(float(probability), text=f"{class_name}: {probability * 100:.2f}%")
