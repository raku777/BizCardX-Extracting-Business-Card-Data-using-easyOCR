import streamlit as st
import easyocr
import cv2
import  numpy as np

reader = easyocr.Reader(['en'])
def process_business_card(image_path):
    # Load the image
    # image = cv2.imread(image_path)
    #image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    image_buffer = uploaded_file.read()
    np_arr = np.frombuffer(image_buffer, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Resize the image 
    image = cv2.resize(image, None, fx=2, fy=2)  # Increase image size by a factor of 2
    
    # Split channels
    b, g, r = cv2.split(image)

    # Apply Adaptive Thresholding on each channel
    _, thresholded_b = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, thresholded_g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, thresholded_r = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Merge channels back into the image
    thresholded = cv2.merge((thresholded_b, thresholded_g, thresholded_r))

    # Denoising
    denoised = cv2.fastNlMeansDenoisingColored(thresholded, None, 10, 10, 7, 21)
    
    
    # Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Extract text from the preprocessed image using EasyOCR
    results = reader.readtext(sharpened, paragraph=True)

    # Process the results
    extracted_info = []
    for result in results:
        text = result[1]  # Extract the recognized text
        extracted_info.append(text)

    return extracted_info


st.write("BizCardX: Extracting Business Card Data with OCR")
uploaded_file = st.file_uploader('Upload you image here',type=['png','jpg','jpeg'])
if uploaded_file is not None:
    st.write(process_business_card(uploaded_file))