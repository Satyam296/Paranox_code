"""
Space Station Safety Object Detection App
==========================================
A simple Streamlit web application for detecting safety equipment in space station images.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# Page configuration
st.set_page_config(
    page_title="Space Station Safety Detector",
    page_icon="🚀",
    layout="wide"
)

# Title and description
st.title("🚀 Space Station Safety Object Detection")
st.markdown("""
This app detects **7 critical safety equipment items** in space station environments:
- 🔵 Oxygen Tank
- 🟢 Nitrogen Tank
- 🔴 First Aid Box
- 🟡 Safety Switch Panel
- 🔥 Fire Extinguisher
- 🚨 Fire Alarm
- 📞 Emergency Phone
""")

# Load model (cached to avoid reloading)
@st.cache_resource
def load_model():
    model_path = "scripts/runs/train/exp4/weights/best.pt"
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        return None
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Run detection
def detect_objects(image, model, conf_threshold=0.25):
    """Run YOLO detection on the image"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Run inference
    results = model(img_array, conf=conf_threshold)
    
    # Get annotated image
    annotated_img = results[0].plot()
    
    # Convert BGR to RGB
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    return annotated_img, results[0]

# Main app
def main():
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence for detection"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📊 Model Info
    - **mAP@0.5:** 54.67%
    - **Precision:** 82.26%
    - **Recall:** 47.16%
    - **Model:** YOLOv8n
    """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.warning("⚠️ Model could not be loaded. Please check the model path.")
        return
    
    # File uploader
    st.markdown("### 📤 Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a space station image to detect safety equipment"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📷 Original Image")
            st.image(image, use_container_width=True)
        
        # Run detection
        with st.spinner("🔍 Detecting objects..."):
            annotated_img, results = detect_objects(image, model, conf_threshold)
        
        with col2:
            st.markdown("#### 🎯 Detection Results")
            st.image(annotated_img, use_container_width=True)
        
        # Display detection statistics
        st.markdown("---")
        st.markdown("### 📊 Detection Summary")
        
        # Count detections by class
        if len(results.boxes) > 0:
            class_names = results.names
            detected_classes = results.boxes.cls.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            
            # Create summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Objects Detected", len(detected_classes))
            
            with col2:
                st.metric("Average Confidence", f"{np.mean(confidences):.2%}")
            
            with col3:
                st.metric("Unique Classes", len(np.unique(detected_classes)))
            
            # Detailed detection table
            st.markdown("#### 📋 Detected Objects")
            
            detection_data = []
            for i, (cls_id, conf) in enumerate(zip(detected_classes, confidences)):
                detection_data.append({
                    "Object #": i + 1,
                    "Class": class_names[int(cls_id)],
                    "Confidence": f"{conf:.2%}"
                })
            
            st.table(detection_data)
            
        else:
            st.info("ℹ️ No objects detected. Try lowering the confidence threshold.")
    
    else:
        # Sample instructions
        st.info("""
        👆 **Upload an image to get started!**
        
        The app will:
        1. Process your image
        2. Detect safety equipment
        3. Draw bounding boxes around detected objects
        4. Show confidence scores for each detection
        """)
        
        # Show example
        st.markdown("---")
        st.markdown("### 🖼️ Example Detections")
        st.markdown("The model can detect objects in various lighting conditions:")
        st.markdown("- ✅ Dark environments")
        st.markdown("- ✅ Bright lighting")
        st.markdown("- ✅ Cluttered scenes")
        st.markdown("- ✅ Partially occluded objects")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>Space Station Challenge: Safety Object Detection #2</strong></p>
        <p>Powered by YOLOv8 | Trained on Falcon Synthetic Data</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
