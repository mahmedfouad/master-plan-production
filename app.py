import streamlit as st
import os
import io
import zipfile
import numpy as np
from PIL import Image

# Set Keras backend to tensorflow before importing Keras
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from huggingface_hub import hf_hub_download  # <--- Import this tool

# 1. Load the model from Hugging Face (cached so it only downloads once)
# 1. Load the model cleanly using Hugging Face's explicit downloader
@st.cache_resource
def load_hf_model():
    # Explicitly downloads your targeted model file from the repository
    model_path = hf_hub_download(
        repo_id="ahmedkindi/master-plan-production", 
        filename="best_mp_model.keras"  # Ensure this filename exactly matches what you uploaded to HF!
    )
    return keras.saving.load_model(model_path)

st.set_page_config(page_title="Image Classification Dashboard", layout="wide")
st.title("📸 Master Plan Classifier")
st.write("Upload your images below to check if they are **Valid** or **Invalid**.")

try:
    model = load_hf_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()
    
# 2. File uploader for multiple images
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    results = []
    valid_images = []
    invalid_images = []

    st.subheader("Processing Images...")
    progress_bar = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        # Read and preprocess the image (Converting to RGB as per training notebook)
        image = Image.open(file).convert('RGB')
        
        # Match training input: Resize to 128x128
        img_resized = image.resize((128, 128)) 
        img_array = keras.utils.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Match training notebook normalization: scale pixels to [0,1]
        img_array = img_array / 255.0

        # Predict (Returns probability map for binary classification)
        prediction = model.predict(img_array, verbose=0)
        raw_prob = float(prediction[0][0])
        
        # Binary Classification Logic: 0 = Invalid, 1 = Valid
        if raw_prob >= 0.7:
            classification = "valid"
            confidence = raw_prob
            valid_images.append(file)
        else:
            classification = "invalid"
            confidence = 1.0 - raw_prob
            invalid_images.append(file)
            
        results.append({
            "Image Name": file.name,
            "Classification": classification,
            "Confidence": f"{confidence:.0%}"
        })
        
        progress_bar.progress((i + 1) / len(uploaded_files))

    # 3. Display Results Table
    st.subheader("📋 Classification Results")
    st.dataframe(results, use_container_width=True)

    # 4. Download Buttons (Zip Creator)
    st.subheader("📦 Download Sorted Images")
    col1, col2 = st.columns(2)

    with col1:
        if valid_images:
            valid_zip = io.BytesIO()
            with zipfile.ZipFile(valid_zip, "w") as z:
                for f in valid_images:
                    z.writestr(f.name, f.getvalue())
            valid_zip.seek(0)
            
            st.download_button(
                label="🟢 Download Valid Images",
                data=valid_zip,
                file_name="valid_images.zip",
                mime="application/zip",
                type="primary"
            )
        else:
            st.info("No valid images found to download.")

    with col2:
        if invalid_images:
            invalid_zip = io.BytesIO()
            with zipfile.ZipFile(invalid_zip, "w") as z:
                for f in invalid_images:
                    z.writestr(f.name, f.getvalue())
            invalid_zip.seek(0)
            
            st.download_button(
                label="🔴 Download Invalid Images",
                data=invalid_zip,
                file_name="invalid_images.zip",
                mime="application/zip",
                type="secondary"
            )
        else:
            st.info("No invalid images found to download.")
