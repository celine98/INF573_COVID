#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 19:49:38 2020

@author: celinehajjar
"""


import cv2
import torch
import torch.utils.data
from PIL import Image
import transforms as T
from itertools import combinations


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_position_from_masks(prediction, focus_len):
    boxes = prediction['boxes']
    labels = prediction['labels']
    scores = prediction['scores']
    masks = prediction['masks']

    if len(labels) == 0:
        return prediction

    positions = []

    total_pixel_height, total_pixel_width = masks[0].shape[1], masks[0].shape[2]

    for index, label in enumerate(labels):
        if label != 1 or scores[index] < 0.75:
            positions.append(torch.tensor([-1.0, -1.0, -1.0]))
            continue
        pixel_height = boxes[index][3] - boxes[index][1]
        pixel_meter_rate = pixel_height / 1.69
        total_height_meter = total_pixel_height / pixel_meter_rate
        distance = total_height_meter / focus_len
        y =(boxes[index][3] + boxes[index][1])/2 - total_pixel_height/2
        x = (boxes[index][0] + boxes[index][2]) / 2 - total_pixel_width / 2
        positions.append(torch.tensor([x/pixel_meter_rate, y/pixel_meter_rate, distance]) )
    prediction['positions'] = positions
    print(positions)
    return prediction

def get_prediction(test_img_path, model, device, focal_len):
    
    test_transform = get_transform(False)
    img, _ = test_transform(Image.open(test_img_path).convert("RGB"), {})
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
    prediction_ = get_position_from_masks(prediction[0], focal_len)    
    
    return prediction_

def human_box_detection(prediction_, frame, safe_distance = 1):
    centers_x = []
    centers_y=[]

    boxes = prediction_['boxes']
    labels = prediction_['labels']
    scores = prediction_['scores']
    positions = prediction_['positions']

    good_indices = []

    for index, label in enumerate(labels):
        if label == 1 and scores[index] >= 0.85:
            box = boxes[index]
            #drawing all boxes with high confidence
            x1,y1,x2,y2 = box[0], box[1], box[2], box[3]
            centers_x+=[(x1+x2)/2]
            centers_y+=[(y1+y2)/2]
            cv2.rectangle(frame, (x1, y1), (x2,y2),(255,0,0), 5)
            cv2.putText(frame,"p{}".format(index), ((x1+x2)/2, y2-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            good_indices+=[index]
            
            
    for (i,j) in combinations(good_indices, 2):
        distance = float(torch.dist(positions[i],positions[j]))
        
        p1 = (centers_x[i],centers_y[i])
        p2 = (centers_x[j], centers_y[j])
        t_x = (p1[0]+p2[0])/2 - 50
        t_y = (p1[1]+p2[1])/2 - 10
        
        if (distance<=safe_distance):
            cv2.line(frame, p1, p2, (0,0,255), 2)
            label = "{:.1f}m".format(distance)
            cv2.putText(frame, label , (t_x,t_y),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            print("Distance between person {} and person {}: {} || NOT SAFE".format(i,j,distance))
        
        else:
            '''
            cv2.line(frame, p1, p2, (0,255,0), 2)
            label = "{:.1f}m".format(distance)
            cv2.putText(frame, label , (t_x,t_y),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        '''
            print("Distance between person {} and person {}: {}".format(i,j,distance))
            
    return 