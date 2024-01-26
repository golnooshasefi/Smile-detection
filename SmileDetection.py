import cv2
import numpy as np
import joblib
import tensorflow as tf

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") 
flag = False
SmileModel = tf.keras.models.load_model("Smile_detection_model/content/SmileDetection_model")
cap=cv2.VideoCapture(0)
vwriter=cv2.VideoWriter('output.wmv',cv2.VideoWriter_fourcc(*'WMV1'),20,(640,480))
while(True):
    ret, frame = cap.read()
    if ret:
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_classifier.detectMultiScale(
            gray_image, 1.1,4)

        for (x, y, w, h) in face:
            img = frame[y:y+h , x:x+w]
            flag = True
        if(flag):
            faceResize = cv2.resize(img, (160, 160))
            label = SmileModel.predict(np.expand_dims(faceResize, axis=0))
            if(label > 0):
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            vwriter.write(frame)
            cv2.imshow('frame', frame)
            cv2.waitKey(5)
    else:
        vwriter.release()
