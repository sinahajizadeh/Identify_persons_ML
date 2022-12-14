import numpy as np
import cv2 as cv



haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

people = ['the names that you want recognize them like part A ']
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')


cap=cv.VideoCapture(0)

while True:
    _,fr=cap.read()

    gray = cv.cvtColor(fr,cv.COLOR_BGR2GRAY)
    

# Detect the face in the image
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h,x:x+w]

        label, confidence =face_recognizer.predict(faces_roi)
        

        cv.putText(fr, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        cv.rectangle(fr, (x,y), (x+w,y+h), (0,255,0), thickness=2)



    if cv.waitKey(20) & 0xFF==ord('d'):

        break    

    cv.imshow('Detected Face', fr)

    cv.waitKey(1)
