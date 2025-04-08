import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os

# Set page layout
st.set_page_config(page_title="Forest Fire Detector", layout="centered")
st.title("🔥 Forest Fire Detection App 🌲")
st.write("Upload an image to check if it shows signs of a forest fire.")

# Function to download and load model
@st.cache_resource
def download_and_load_model():
    model_path = "forest_fire_detector.h5"
    if not os.path.exists(model_path):
        file_id = "1sQjGhSczD1sIRTMM-OTuk9qcIbddUHwd"  # ✅ Your actual file ID
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    return load_model(model_path)

model = download_and_load_model()

# Class labels used during training
class_names = ['fire', 'no_fire']

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    predicted_label = class_names[class_index]
    confidence = prediction[0][class_index] * 100

    # Result display
    st.write(f"### Prediction: `{predicted_label.upper()}`")
    st.write(f"Confidence: `{confidence:.2f}%`")

    if predicted_label == 'fire':
        st.error("⚠️ Fire Detected! Take action immediately!")
    else:
        st.success("✅ No Fire Detected. Environment seems safe.")
