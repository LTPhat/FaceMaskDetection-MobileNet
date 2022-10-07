from unicodedata import name
from tensorflow import keras
from keras.preprocessing import image
from PIL import Image
import streamlit as st
import numpy as np
import base64
import warnings
warnings.filterwarnings('ignore')
import os
import cv2
from keras.utils import img_to_array, load_img
<<<<<<< HEAD
classifier =keras.models.load_model('D:\FaceMaskDetection-MobileNet-\Face_mask.h5')
=======
classifier =keras.models.load_model("D:\FaceMaskDetection-MobileNet-\Face_mask.h5")
>>>>>>> 3e82472239d603e6b9d6d986f769e6e4f8fb2d57
classes = ['mask_weared_incorrect','with_mask','without_mask']
face_classifier = cv2.CascadeClassifier(r"C:\Users\HP\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
# Read file
label = ""
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
# Set background for local web
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
<<<<<<< HEAD
set_background('D:\FaceMaskDetection-MobileNet-\web_background.png')
=======
set_background('D:\FaceMaskDetection-MobileNet-\Back1.jpg')
>>>>>>> 3e82472239d603e6b9d6d986f769e6e4f8fb2d57
def predict_img(img):
    # Detect face
    global label
    img = np.array(img)
    faces = face_classifier.detectMultiScale(img, 1.1, 4)
    for (x,y,w,h) in faces:
        # Crop input image
        # [y: y + h, x: x + w]
        roi_img = img[y - 20: y + h + 20, x - 40: x + w + 20]
        # Resize to CNN input shape
        roi_img = cv2.resize(roi_img,(128,128),interpolation=cv2.INTER_AREA)
        # Successfully detect image
        if np.sum([roi_img])!=0:
            roi = roi_img.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            prediction = classifier.predict(roi)[0]
            label = classes[prediction.argmax()]
            result = str(label + ": " + str(np.round(prediction[prediction.argmax()], 3)))
        if label == "with_mask":
            label_position = (x, y)
            # Create rectangle bounding box
            cv2.rectangle(img, (x-20, y+20), (x + w-20, y + h+20), (0, 100, 0), 3)
            cv2.putText(img, result, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        elif label == 'mask_weared_incorrect':
            label_position = (x-150, y)
            cv2.rectangle(img, (x-20, y+20), (x + w-20, y + h+20), (0, 128, 128), 3)
            cv2.putText(img, result, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        elif label == 'without_mask':
            label_position = (x-50, y)
            cv2.rectangle(img, (x-20, y+20), (x + w-20, y + h+20), (0, 0, 139), 3)
            cv2.putText(img, result, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.putText(img, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    return img, label
# def predict_class(img):
#     global label
#     img = np.array(img)
#     faces = face_classifier.detectMultiScale(img, 1.1, 4)
#     if faces is not None:
#         for (x, y, w, h) in faces:
#             # Crop input image
#             # [y: y + h, x: x + w]
#             roi_img = img[y - 20: y + h + 20, x - 20: x + w + 20]
#             roi_img = cv2.resize(roi_img, (128, 128), interpolation=cv2.INTER_AREA)
#             roi = roi_img.astype('float') / 255.0
#             roi = np.expand_dims(roi, axis=0)
#             prediction = classifier.predict(roi)[0]
#             label = classes[prediction.argmax()]
#     else:
#         st.write("Can't detect face")
#     return label
def main():
    st.markdown("<h1 style='text-align:center; color: black;'>Welcome To Face Mask Detector</h1>",
                unsafe_allow_html=True)
    html_temp = """
       <div style="background-color: brown ;padding:5px">
       <h3 style="color:black;text-align:center; font-size: 15 px"> Click the below button to upload image.</h3>
       </div>
       """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown("")
    uploaded_file = st.file_uploader("Choose image file", accept_multiple_files=False)
    if uploaded_file is not None:
        st.write("File uploaded:", uploaded_file.name)
        show_img = load_img(uploaded_file,target_size=(500,500))
        st.image(show_img, caption= "Original image uploaded")
    if st.button("Predict"):
        result_img,result_label = predict_img(show_img)
        st.image(result_img, caption='Result')
        st.success('Predict: {}'.format(str(result_label)))
if __name__=='__main__':
    main()