#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:37:04 2019

@author: joe
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
##############################################################################
img = cv2.imread('/home/joe/Desktop/intensity_enhancement/calibrated/2019_0711_214053_142_CALIBRATED.JPG')
n,c,o =  cv2.split(img)
a1 = (n-c)
t2,a2 = cv2.threshold(a1,50,1,cv2.THRESH_BINARY)

#mask = cv2.merge((a2, a2, a2))
#img1 = img*mask

#white flower
t3,a3 = cv2.threshold(c,190,1,cv2.THRESH_BINARY)
kernel = np.ones((5,5),np.uint8)
a3 = cv2.morphologyEx(a3, cv2.MORPH_OPEN, kernel)
a3 = cv2.dilate(a3,kernel,iterations = 4)
#mask1 = cv2.merge((a3, a3, a3))
#img2 = img*mask1

#disease
# Convert RGB to HSV
hsv = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
h,s,v = cv2.split(hsv)   
# Convert RGB to luv
luv = cv2.cvtColor(img1, cv2.COLOR_RGB2Luv)
l,u,v1 = cv2.split(luv)   
a4 = n-o+v1+h
t4,a4 = cv2.threshold(a4,190,1,cv2.THRESH_BINARY)
a4 = cv2.morphologyEx(a4, cv2.MORPH_OPEN, kernel)
a4 = cv2.erode(a4,kernel,iterations = 1)
a4 = cv2.dilate(a4,kernel,iterations = 5)

#mask2 = cv2.merge((a4, a4, a4))
#img3 = img*mask2

#a2 = a2 - a4
#merge
img_new = cv2.merge((a3*255, a2*255, a4*255))
#plt.hist(a4.ravel(),256,[0,256]); plt.show()

plt.imshow(img_new);plt.show()

cv2.imwrite('/home/joe/Desktop/test.jpg', img_new)