#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:36:20 2020

@author: joe
"""

import matplotlib.pyplot as plt
#import os
import cv2
import numpy as np
from keras.models import model_from_json,Model
import keras.backend as K
##############################################################################
# load json and create model
json_file = open('/home/joe/Desktop/sgan/sgan_test/model_resnet_64b_10ki_noBN_400lat_.00001_.7_5class/saved_model/mnist_sgan_discriminator.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.summary()
# load weights into new model
loaded_model.load_weights("/home/joe/Desktop/sgan/sgan_test/model_resnet_64b_10ki_noBN_400lat_.00001_.7_5class/saved_model/mnist_sgan_discriminator_weights.hdf5")
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#print
# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
# org 
org = (50, 50) 
# fontScale 
fontScale = 1
# Blue color in BGR 
color = (255, 0, 0) 
# Line thickness of 2 px 
thickness = 2
#patches
w1 = 200
h1 = 200
# Print the files
x = 0
y = 0
# Load an color image in colour
img_test = cv2.imread('/home/joe/Desktop/sgan/train/clustering/kmeans50.jpg')
#img_test = cv2.imread('/home/joe/Desktop/sgan/train/ocn/ocn51.jpg')
#patches
for i in range(0,20):
    for j in range(0,15):
        cropped_img = img_test[y:y+h1, x:x+w1].copy()
        cropped_img = cropped_img.reshape(1,200,200,3)
        valid, label = loaded_model.predict(cropped_img)
        #print("valid:",valid,"label:",label)
        if valid >= 0.1:
            if label[0][0] >= .5: 
                # Using cv2.putText() method 
                #img_test = cv2.putText(img_test, 'potato', (x+30,y+90), font, fontScale, (0,255,0), thickness, cv2.LINE_AA)
                mask = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                mask = cv2.merge((mask,mask,100+mask))
                cropped_img = mask
            elif label[0][1] >= .5: 
                # Using cv2.putText() method 
                img_test = cv2.putText(img_test, 'disease', (x+30,y+90), font, fontScale, (0,0,255), 3, cv2.LINE_AA)
            elif label[0][2] >= .5: 
                # Using cv2.putText() method 
                img_test = cv2.putText(img_test, 'grass', (x+30,y+90), font, fontScale, (250,250,0), thickness, cv2.LINE_AA)
            #elif label[0][3] >= .5: 
                # Using cv2.putText() method 
                #img_test = cv2.putText(img_test, 'weed3', (x+30,y+90), font, fontScale, (250,250,150), thickness, cv2.LINE_AA)
            elif label[0][4] >= .5: 
                # Using cv2.putText() method 
                img_test = cv2.putText(img_test, 'pumpkin', (x+30,y+90), font, fontScale, (0,250,200), thickness, cv2.LINE_AA)
            elif label[0][5] >= .5:
                #Using cv2.putText() method 
                img_test = cv2.putText(img_test, 'unknown', (x+30,y+90), font, fontScale, (255,255,255), thickness, cv2.LINE_AA)
                
        y = y+h1
    x = x+w1
    y = 0

cv2.imwrite('/home/joe/Desktop/sgan_test_model_resnet_64b_10ki_noBN_400lat_.00001_.7_5class.png', img_test)
plt.imshow(img_test)
plt.show()
print ("********************testing complete****************")
