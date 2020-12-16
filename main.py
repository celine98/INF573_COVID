#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:34:08 2020

@author: celinehajjar
"""


import cv2
import os
import torch
import torch.utils.data

import argparse


from tensorflow.keras.models import load_model
from human_detection import get_transform, get_position_from_masks, get_prediction, human_box_detection
from face_detection import face_detection
from mask_detection import mask_detection
    
def load_models(device):
    #person detection
    model = torch.load("./models/mask_rcnn_rfined-full.pt", map_location=torch.device('cpu') )
    model.to(device)

    #mask detection
    cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    #model_masks = load_model("./mask_detection/mask_recog.h5")
    model_masks = load_model("./mask_detector.model")
    
    return model, faceCascade, model_masks


def run():
    
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-i", "--image", required=True,
		help="path to test image")
    
    ap.add_argument("-sd", "--safe_distance", type=int, 
                    default=1, 
                    help="Desired safe distance in meters (default = 1m)")
    
    ap.add_argument("-fl", "--focal_len", type = float, 
                    default = 0.965, help = "Focal Length if known (default = 0.965)")
    
    args = vars(ap.parse_args())
    
    test_img_path = args["image"]
    safe_distance = args["safe_distance"]
    focal_len = args["focal_len"]
    
    #loading models
    print("-----------------------------------------------")
    print("Loading Models")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model, faceCascade, model_masks = load_models(device)
    
    print("-----------------------------------------------")
    print("Looking for humans")
    
    frame = cv2.imread(test_img_path)
    
    #Human detection
    prediction_ = get_prediction(test_img_path, model, device, focal_len)
    
    #box annotation
    human_box_detection(prediction_, frame, safe_distance)
    
    print("-----------------------------------------------")
    print("Looking for faces and masks")
    
    #faces detection  
    faces = face_detection(prediction_, frame, faceCascade)
    
    #mask detection
    mask_detection(faces, frame, model_masks)

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result",frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    '''
    test_img_path = os.path.join("./X/test_masks_2-2_4.jpg")
    safe_distance = 1
    '''
    run()
