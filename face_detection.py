#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 19:54:27 2020

@author: celinehajjar
"""


import cv2


def face_detection(prediction_, frame, faceCascade):
    
    boxes = prediction_['boxes']
    labels = prediction_['labels']
    scores = prediction_['scores']
    
    faces = {}
    
    for index, label in enumerate(labels):
        if label == 1 and scores[index] >= 0.85:
            box = boxes[index]
            #drawing all boxes with high confidence
            x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(frame, (x1, y1), (x2,y2),(255,0,0), 5)
            person_frame = frame[y1:y2, x1:x2] 
            gray = cv2.cvtColor(person_frame, cv2.COLOR_BGR2GRAY)
            faces_ = faceCascade.detectMultiScale3(gray,
                                                 scaleFactor=1.05,
                                                 minNeighbors=10,
                                                 minSize=(60, 60),
                                                 flags=cv2.CASCADE_SCALE_IMAGE,
                                                 outputRejectLevels = True)
            faces_transformed = []
            for (i, (x, y, w, h)) in enumerate(faces_[0]):
                x = x + x1
                y = y + y1
                print("Face detected in person ", index)
                faces_transformed.append((x,y,w,h))
                
            if len(faces_transformed)==0:
                w = int((x2 - x1)/2)
                x = x1 + int((x2-x1-w)/2)
                y = y1
                h = int((y2-y1)/6)
                faces_transformed.append((x,y,w,h))
                print("No face detected in person {}, one face added".format(index) )
               
            faces[str(index)] = faces_transformed
            
            
    
    return faces