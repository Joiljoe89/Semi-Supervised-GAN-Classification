#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:50:09 2019

@author: joe
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
##############################################################################
img_k_14 = cv2.imread('/home/joe/Desktop/ss_field/28_10_19/kernel_14_4/normal/POCN00067.jpg')
img_k_3 = cv2.imread('/home/joe/Desktop/ss_field/28_10_19/kernal_3_2/Processed_2/P72500071.jpg')
img_pi_8 = cv2.imread('/home/joe/Desktop/ss_field/28_10_19/Pi_cam_8mp/image2019-10-24-16-52-57.jpg')
img_s_3n = cv2.imread('/home/joe/Desktop/ss_field/28_10_19/survey3_N/2019_1028_122221_019.JPG')
img_s_rgn_3w = cv2.imread('/home/joe/Desktop/ss_field/28_10_19/survey3_rgn_W/2019_1027_225231_106.JPG')
img_s_3w = cv2.imread('/home/joe/Desktop/ss_field/28_10_19/survey3_W/2018_0427_152027_019.JPG')
o,c,n = cv2.split(img_s_3w)
r,g,n1 = cv2.split(img_s_rgn_3w)
o1,c1,n2 =  cv2.split(img_k_14)


'''
a = c1 -o1
kernel = np.ones((5,5),np.uint8)

dilation = cv2.dilate(a,kernel,iterations = 1)
erosion = cv2.erode(a,kernel,iterations = 1)
opening = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel)

kernel1 = np.ones((15,15),np.uint8)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel1)

kernel2 = np.ones((11,11),np.uint8)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)

erosion = cv2.erode(opening,kernel2,iterations = 3)
dilation = cv2.dilate(erosion,kernel1,iterations = 5)

gray_filtered = cv2.inRange(gray, 190, 255)
rgb_filtered = cv2.inRange(gray, (190, 190, 190), (255, 255, 255))

plt.imshow(dilation)
plt.show()
'''