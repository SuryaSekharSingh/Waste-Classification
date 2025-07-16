import streamlit as st
from PIL import Image
import numpy as np
import time
import webbrowser
import tensorflow as tf

# Set page config
st.set_page_config(
    page_title="Waste Classification System",
    page_icon="🗑️",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        text-align: center;
        color: #2e8b57;
        padding-bottom: 20px;
    }
    .upload-box {
        border: 2px dashed #2e8b57;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        background-color: #f8f9fa;
    }
    .result-box {
        border: 2px solid #2e8b57;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        background-color: #f0fff0;
    }
    .info-box {
        border-left: 4px solid #2e8b57;
        padding: 15px;
        background-color: #f8f9fa;
        margin-top: 20px;
    }
    .video-container {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
    }
    .video-thumbnail {
        width: 48%;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .video-thumbnail:hover {
        transform: scale(1.02);
    }
    .stProgress > div > div > div > div {
        background-color: #2e8b57;
    }
</style>
""", unsafe_allow_html=True)

# Recycling information (moved from util.py)
CLASS_INFO = {
    "Batteries": {
        "info": "Battery recycling is a recycling activity that aims to reduce the number of batteries being disposed as municipal solid waste. Batteries contain a number of heavy metals and toxic chemicals...",
        "videos": ["4XOAGNzWvqY", "oKFOqMZmuA8"]
    },
    "Clothes": {
        "info": "Textile recycling is the process of recovering fiber, yarn or fabric and reprocessing the textile material into useful products...",
        "videos": ["Bhi7S06pwv4", "IHPBJySIXZw"]
    },
    # Add all other classes in the same format
    # ...
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('classifyWaste.keras')

def display_video_thumbnails(video_id1, video_id2):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<a href="https://www.youtube.com/watch?v={video_id1}" target="_blank">'
            f'<img class="video-thumbnail" src="https://img.youtube.com/vi/{video_id1}/0.jpg" alt="Video 1">'
            '</a>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<a href="https://www.youtube.com/watch?v={video_id2}" target="_blank">'
            f'<img class="video-thumbnail" src="https://img.youtube.com/vi/{video_id2}/0.jpg" alt="Video 2">'
            '</a>',
            unsafe_allow_html=True
        )

def classify_waste(image_path, model):
    # Load and preprocess image
    test_image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)
    
    # Make prediction
    predicted_array = model.predict(test_image)
    predicted_class = np.argmax(predicted_array)
    class_names = list(CLASS_INFO.keys())
    waste_type = class_names[predicted_class]
    
    return waste_type, CLASS_INFO[waste_type]["info"], CLASS_INFO[waste_type]["videos"][0], CLASS_INFO[waste_type]["videos"][1]

def main():
    # Load model
    model = load_model()
    
    # Header
    st.markdown("<h1 class='header'>♻️ Classify Your Waste Material</h1>", unsafe_allow_html=True)
    st.markdown("Click the image upload icon below to upload an image.")
    
    # File uploader
    with st.container():
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            " ",  # Empty label
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_upload.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text(f"Analyzing... {i+1}%")
            time.sleep(0.02)
        
        # Make prediction
        waste_type, info, video1, video2 = classify_waste("temp_upload.jpg", model)
        
        # Clear progress
        progress_bar.empty()
        status_text.empty()
        
        # Show results
        st.markdown(f"<h2 style='text-align: center;'>Waste classified as {waste_type}</h2>", unsafe_allow_html=True)
        
        # Result box
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown(f"**How to recycle {waste_type}:**")
        st.markdown(info, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Video thumbnails
        st.markdown("### Recycling Tutorial Videos")
        display_video_thumbnails(video1, video2)

if __name__ == "__main__":
    main()