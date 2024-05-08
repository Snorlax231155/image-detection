#Take input image for object detection.

import streamlit as st
from views import model_information,header,sidebar
from predict import predict
def main():
    with open("styles.css", "r") as source_style:
        st.markdown(f"<style>{source_style.read()}</style>",
                    unsafe_allow_html=True)
    header.main()
    selected = sidebar.main()

    if selected == "Predict":
        # Image
        upload_img_file = st.sidebar.file_uploader(
            'Upload Image', type=['jpg', 'jpeg', 'png'])
        if upload_img_file is not None:
            image_bytes = predict(upload_img_file)
            st.image(image_bytes, caption='Predicted Image',
                     use_column_width=True)
    elif selected == "Model Information":
        model_information.main()
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
#Classify the Objects
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
model = YOLO('best.pt')

def predict(upload_img_file):
    file_bytes = np.asarray(
        bytearray(upload_img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    prediction = model.predict(img)
    res_plotted = prediction[0].plot()
    image_pil = Image.fromarray(res_plotted)
    image_bytes = io.BytesIO()
    image_pil.save(image_bytes, format='PNG')

    return image_bytes
