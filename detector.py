import cv2
import os
import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Flatten,MaxPool2D,Conv2D,Dropout,MaxPooling2D,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical, plot_model
from keras import models, layers, regularizers
from keras.preprocessing import image
from keras.utils import img_to_array


face_classifier = cv2.CascadeClassifier(r"C:\Users\HP\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
classifier =keras.models.load_model('E:\FaceMaskDetection\Face_mask.h5')

classes = ['mask_weared_incorrect','with_mask','without_mask']

cap = cv2.VideoCapture(0)



while True:
    #Capture video
    _, frame = cap.read()
    labels = []
    # Convert color channel
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #Detect face
    faces = face_classifier.detectMultiScale(img,1.2,4)

    for (x,y,w,h) in faces:
        #Create rectangle bounding box
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #Crop input image
        roi_img = img[y:y+h,x:x+w]
        #Resize to CNN input shape
        roi_img = cv2.resize(roi_img,(128,128),interpolation=cv2.INTER_AREA)
        # Successfully detect image
        if np.sum([roi_img])!=0:
            roi = roi_img.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)
            prediction = classifier.predict(roi)[0]
            label=classes[prediction.argmax()]
            label_position = (x,y)
            result = str(label + ": "+ str(np.round(prediction[prediction.argmax()],3)))
            if label == "with_mask":
                # Create rectangle bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,100,0), 3)
                cv2.putText(frame, result, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
            elif label == 'mask_weared_incorrect':
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,128,128), 3)
                cv2.putText(frame, result, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,139), 3)
                cv2.putText(frame, result, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        # No detect
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    cv2.imshow('Face Mask Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()