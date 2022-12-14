
import os

import cv2 as cv 
import numpy as np



people=[' give the names thah you want to recognizing them (the names shoude exist in your path)']

myface=cv.CascadeClassifier('haarcascade_frontalface_default.xml')

D=r'imgtest'




feature=[]
Labels=[]

def creat_train():
    for person in people:
        path=os.path.join(D,person)
        LabeL=people.index(person)


        

        for img in os.listdir(path):
            img_path=os.path.join(path,img)

            
            
            img_array=cv.imread(img_path)
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            faces_react=myface.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

            for (x,y,w,h) in faces_react:
                faces_r=gray[y:y+h,x:x+w]
                feature.append(faces_r)
                Labels.append(LabeL)

        
creat_train()

feature=np.array(feature,dtype='object')
Labels=np.array(Labels)


face_recognizer=cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(feature,Labels)

np.save('features.npy', feature)
np.save('labels.npy', Labels)
face_recognizer.save('face_trained.yml')

            
