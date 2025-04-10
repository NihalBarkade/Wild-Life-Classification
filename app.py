import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np

# Load the trained model
model = load_model("best_model.h5")

# Define the class names (same as during training)
class_names = [
    'Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant',
    'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra'
]

# Function to preprocess the image
def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Streamlit App
st.set_page_config(page_title="Animal Classifier", layout="centered")
st.title("🐾 Animal Image Classifier")
st.markdown("""
Welcome to the **Animal Image Classifier**!   

This deep learning model uses **MobileNetV2 + Transfer Learning** to predict the animal in a given image.

**📊 Training Summary:**
- **Model Architecture:** MobileNetV2 (pretrained on ImageNet)
- **Dataset Size:** 3000 images (with 15 classes)
- **Image Size:** 224 × 224 × 3
- **Training Accuracy:** ~99%
- **Validation Accuracy:** ~94%
- **Test Accuracy:** ~87%

Upload an image of an animal and the model will tell you what it is — along with how confident it is in the prediction. 🔍
""")

uploaded_file = st.file_uploader("Upload an animal image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Predicting..."):
        img_array = preprocess_image(uploaded_file)
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = round(100 * np.max(prediction), 2)

    st.success(f"🧠 Prediction: **{predicted_class}**")
    st.info(f"Confidence: **{confidence}%**")

# Footer Section
st.markdown("---")
st.markdown("Developed by **Nihal Barkade** as part of an internship project on image classification using deep learning.")