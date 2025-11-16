import streamlit as st
from model_helper import predict

st.title("Velocity Flow Regimn")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image_path = "temp_upload.png"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    prediction = predict(image_path)
    st.success(f"Predicted Class: {prediction}")

