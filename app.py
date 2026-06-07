import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model("csr_model.keras")
st.write("App Started")

st.title("CSR Detection from OCT Scan")

uploaded_file = st.file_uploader(
    "Upload an OCT Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image")

    image = image.resize((224,224))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    st.write(f"Raw prediction value: `{prediction:.6f}`")

    # OCTID_CSR=0, OCTID_NORMAL=1 (alphabetical class order)
    # So low prediction = CSR, high prediction = Normal
    if prediction < 0.5:
        st.error(f"CSR Detected ({(1-prediction)*100:.2f}%)")
    else:
        st.success(f"Normal Eye ({prediction*100:.2f}%)") 

