# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 18:58:12 2020

@Disclaimer- The dataset and Images are obtained from Public Git reposetories and Tech forums.
"""

import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
from keras.models import load_model
from win32com.client import Dispatch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def speak(str):
	speak = Dispatch(("SAPI.SpVoice"))
	speak.Speak(str)

video_capture = cv2.VideoCapture(0)

image1 = face_recognition.load_image_file(os.path.abspath("EmployeeImageDataset/without mask/me2.jpg"))
image1_face_encoding = face_recognition.face_encodings(image1)[0]

image2 = face_recognition.load_image_file(os.path.abspath("EmployeeImageDataset/without mask/me.jpg"))
image2_face_encoding = face_recognition.face_encodings(image2)[0]

known_face_encodings = [
    image1_face_encoding,
    image2_face_encoding
]
known_face_names = [
    "Rina Jadav ",
    "Sanjay Jadav "
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Write Offender Data to the file
def markAttendance(name):
	with open('EmployeeViolationData.csv','a+', buffering=1) as f:
		myDataList = f.readlines()
		nameList = []
		for line in myDataList:
			entry = line.split(',')
			nameList.append(entry[0])
		if name not in nameList:
			now = datetime.now()
			dtString = now.strftime("%x %X")
			f.writelines(f'\n{name} #{dtString}')

# Mask - UnMask Logic
model = load_model('model-017.model')
face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#source=cv2.VideoCapture(0)
labels_dict={0:' MASK Compliant',1:' is not Wearing MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}            

while True:
    
    
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    print(rgb_small_frame)
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame
    print ("Face detected -- {}".format(face_names))
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        #cv2.putText(frame, name + labels_dict[label], (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, name , (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
				
   # cv2.imshow('Video', frame)
   
    ret,img=video_capture.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.3,5)  

    for x,y,w,h in faces:
    
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        #cv2.putText(img, "{}".format(face_names) + labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.putText(img,( "{}".format(face_names) if len(face_names) != 0  else "" ) + labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255),2)
        
    cv2.imshow('LIVE',img )
    
    if len(face_names) != 0:
        markAttendance("{}".format(face_names) + labels_dict[label])
        speak ("{}".format(face_names) + "ODC access is blocked. Please wear a Mask.")
        
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()