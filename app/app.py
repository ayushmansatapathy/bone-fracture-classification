import streamlit as st
import sys
import os

# Add src folder to path
sys.path.append('../src')

from predict import load_trained_model, predict_single_image

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "../models/final_model.h5"

# Load model
model = load_trained_model(MODEL_PATH)

# Class labels (IMPORTANT: must match training)
class_labels = [
    'avulsion fracture',
    'comminuted fracture',
    'fracture dislocation',
    'greenstick fracture',
    'hairline fracture',
    'impacted fracture',
    'longitudinal fracture',
    'oblique fracture',
    'spiral fracture',
    'transverse fracture'
]

# -------------------------------
# UI
# -------------------------------
st.title("🦴 Bone Fracture Classifier")
st.write("Upload an X-ray image to predict fracture type")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = os.path.join("temp_image.jpg")
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show image
    st.image(temp_path, caption="Uploaded Image", use_column_width=True)

    # Predict
    result = predict_single_image(model, temp_path, class_labels)

    # Display result
    st.subheader("Prediction:")
    st.write(f"🩺 **Type:** {result['prediction']}")
    st.write(f"📊 **Confidence:** {result['confidence']:.2f}")