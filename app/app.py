# app/app.py — add this at the top
import sys, pathlib
project_root = pathlib.Path(__file__).resolve().parents[1]   # one level up from app/
sys.path.append(str(project_root))


import streamlit as st
import numpy as np
import cv2
from PIL import Image
from src.detect_and_ocr import preprocess_image, detect_digits, reconstruct_reading, visualize

st.set_page_config(page_title="Medical Display Reader", layout="wide")

CUSTOM_CSS = r"""
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, .stMarkdown h1 {
    color: #07A0C3;
    font-weight: 700;
    letter-spacing: -0.5px;
}
h2, .stMarkdown h2 {
    color: #86cfe0;
}
button[kind="primary"], .stButton>button {
    background: linear-gradient(180deg, #07A0C3 0%, #086788 100%) !important;
    border: none !important;
    box-shadow: 0 6px 18px rgba(6,103,120,0.18) !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    padding: 0.6rem 1rem !important;
    font-weight: 600 !important;
}
button[kind="secondary"], .stButton>button[aria-pressed] {
    background: transparent !important;
    color: #07A0C3 !important;
    border: 1px solid rgba(7,160,195,0.18) !important;
}
.stFileUploader>div, .stFileUploader {
    background: linear-gradient(90deg, rgba(7,160,195,0.06), rgba(8,103,136,0.03));
    border: 1px dashed rgba(7,160,195,0.18);
    border-radius: 12px;
    padding: 0.9rem;
}
.result-card {
    background: #0b1b1e;
    border-left: 4px solid #07A0C3;
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 8px 24px rgba(3,15,18,0.6);
}
.reading-badge {
    background: linear-gradient(90deg,#086788,#07A0C3);
    color: #fff;
    padding: 0.7rem 1.2rem;
    border-radius: 10px;
    font-weight: 700;
    display: inline-block;
}
.calibrated-image {
    border-radius: 12px;
    box-shadow: 0 16px 40px rgba(2,6,8,0.6);
    border: 4px solid rgba(7,160,195,0.06);
}
.img-legend {
    color: #e6f6f8;
    background: rgba(3,15,18,0.45);
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    font-weight: 600;
}
.confidence {
    color: #086788 !important;
    font-weight: 700;
    font-size: 0.8rem !important;
}
@media (max-width: 900px) {
    .reading-badge { font-size: 0.9rem; padding: 0.6rem 1rem; }
    .calibrated-image { width: 100% !important; height: auto !important; }
}
"""
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)


st.title("Medical Display Digit Reader")
st.write("Upload an image of a medical display (BP monitor, glucometer, oximeter).")

img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if img_file:
    pil_img = Image.open(img_file).convert("RGB")
    img = np.array(pil_img)
    st.image(img, caption="Uploaded Image")

    if st.button("Run Detection"):
        img_p = preprocess_image(img)
        boxes = detect_digits(img_p)
        reading = reconstruct_reading(img_p, boxes)
        vis = visualize(img_p, boxes, reading)

        st.image(vis[:, :, ::-1], caption=f"Detected Reading: {reading}")
        st.success(f"Reading: {reading}")
