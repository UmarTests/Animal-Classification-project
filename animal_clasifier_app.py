import streamlit as st
from PIL import Image
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern

# Load model
assets = joblib.load(r'C:\Users\mohdq\OneDrive\Desktop\internship projects\Animal_classi.pro\Animal Classification\animal_classifier.joblib')
clf = assets['model']
le = assets['encoder']

st.title("Animal Classifier")
uploaded_file = st.file_uploader("Upload animal image", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((64, 64))
    img = np.array(img)
    
    # Feature extraction
    hog_feat = hog(img, orientations=6, pixels_per_cell=(16,16),
                 cells_per_block=(1,1), channel_axis=-1)
    gray = (np.mean(img, axis=2) * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray, P=8, R=1)
    lbp_hist = np.histogram(lbp, bins=10, range=(0, 255))[0]
    features = np.concatenate([hog_feat, lbp_hist])
    
    # Prediction with confidence
    probs = clf.predict_proba([features])[0]
    pred = le.inverse_transform([np.argmax(probs)])
    confidence = max(probs)
    
    # Display
    st.image(img, caption=f"Prediction: {pred[0]} (Confidence: {confidence:.0%})", 
             use_container_width=True)
    
    if confidence < 0.7:
        st.warning("Low confidence - try a clearer image")
        st.write("Other possibilities:", 
                le.classes_[np.argsort(probs)[-3:][::-1][1:]])