#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:42:45 2019

@author: joe
"""

import cv2
import os
import matplotlib.pyplot as plt


import numpy as np


#img_in = cv2.imread('/home/joe/Desktop/sgan/train/clustering/kmeans60.jpg')

img_in = cv2.imread('/home/joe/Desktop/sgan/train/ocn/ocn60.jpg')
# Convert RGB to HSV
hsv = cv2.cvtColor(img_in, cv2.COLOR_RGB2HSV)
# Convert RGB to luv
luv = cv2.cvtColor(img_in, cv2.COLOR_RGB2Luv)
# Convert RGB to lab
lab = cv2.cvtColor(img_in, cv2.COLOR_RGB2Lab)
# Convert RGB to YCrCb
ycrcb = cv2.cvtColor(img_in, cv2.COLOR_RGB2YCrCb)
#img_in = cv2.merge((img_in,hsv,luv))

plt.imshow(hsv[:,:,1]-hsv[:,:,2])
#cv2.imwrite('/home/joe/Desktop/test_luv.jpg', luv)
plt.show()

a = hsv[:,:,1]-hsv[:,:,2]
#a = 255-a
t,a = cv2.threshold(a,200,250,cv2.THRESH_TOZERO_INV)
plt.imshow(a)

plt.imshow(ycrcb[:,:,0]); plt.show()
#plt.savefig("/home/joe/Desktop/test.png")
plt.hist(a.ravel(),256,[0,256]); plt.show()

img_out = cv2.merge((hsv[:,:,2],lab[:,:,1],ycrcb[:,:,2]))
cv2.imwrite('/home/joe/Desktop/test.jpg', img_out)



n1 = img_out.reshape((-1,1))
n1 = np.float32(n1)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5
ret,label,center=cv2.kmeans(n1,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img_out.shape))
plt.imshow(res2[:,:,2])
plt.hist(res2[:,:,1].ravel(),256,[0,256]); plt.show()
#cv2.imshow('res2',res2)
t3,a3 = cv2.threshold(res2,150,1,cv2.THRESH_BINARY)
mask = cv2.merge((a3, a3, a3))
img1 = img_in*mask
cv2.imwrite('/home/joe/Desktop/test1.jpg', res2)