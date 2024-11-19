#model = tf.keras.models.load_model("C:\Users\Sarnika\Downloads\best_model.keras")
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf  # or torch if using PyTorch

# Load your pre-trained Siamese model
model = tf.keras.models.load_model(r"C:\Users\Sarnika\OneDrive\Desktop\sem 5\cap\best_model (1).h5")
print("ggggggggggggggg")

st.title("Breast Cancer Classification Using Siamese Network")

# Upload two images
uploaded_file1 = st.file_uploader("Upload the first mammogram image", type=["jpg", "png"])
uploaded_file2 = st.file_uploader("Upload the second mammogram image", type=["jpg", "png"])

if uploaded_file1 and uploaded_file2:
    image1 = Image.open(uploaded_file1).convert("RGB")
    image2 = Image.open(uploaded_file2).convert("RGB")
    
    st.image([image1, image2], caption=["Image 1", "Image 2"], width=150)
    
    # Preprocess images as per your model's requirements
    image1 = np.array(image1.resize((224, 224))) / 255.0
    image2 = np.array(image2.resize((224, 224))) / 255.0
    
    # Predict similarity
    prediction = model.predict([np.expand_dims(image1, axis=0), np.expand_dims(image2, axis=0)])
    
    # Display result
    if prediction < 0.5:  # threshold based on training
        st.write("The images are similar: Likely Breast Cancer")
    else:
        st.write("The images are dissimilar: Likely Not Breast Cancer")
