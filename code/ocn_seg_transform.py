#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 14:16:04 2019

@author: joe
"""

import cv2
import os
import numpy as np

##############################################################################
'''
    For the given path, get the List of all files in the directory tree 
'''
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles
##############################################################################
'''
# Load an color image in grayscale, 1, 0 or -1 ->colour,grayscale, 
#with same alpha value respectively
img = cv2.imread('messi5.jpg',0)
'''

# training images
dirName = '/home/joe/Desktop/intensity_enhancement/calibrated';
# Get the list of all files in directory tree at given path
listOfFiles = getListOfFiles(dirName)

k = 0
# Print the files
for elem in listOfFiles:
   
    print(elem[51::])
    # Load an color image in colour
    img = cv2.imread('/home/joe/Desktop/intensity_enhancement/calibrated/'+elem[51::])
    cv2.imwrite('/home/joe/Desktop/sgan/train/ocn/ocn%s.jpg' %k, img)
    #img = cv2.imread('C:/Users/Pinnacle/Desktop/2019_0711_214045_134_CALIBRATED.JPG')
    n,c,o =  cv2.split(img)
    a1 = (n-c)
    t2,a2 = cv2.threshold(a1,50,1,cv2.THRESH_BINARY)
    mask = cv2.merge((a2, a2, a2))
    img1 = img*mask
    #white flower
    t3,a3 = cv2.threshold(c,190,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    a3 = cv2.morphologyEx(a3, cv2.MORPH_OPEN, kernel)
    a3 = cv2.dilate(a3,kernel,iterations = 4)
    
    #disease
    # Convert RGB to HSV
    hsv = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
    h,s,v = cv2.split(hsv)   
    # Convert RGB to luv
    luv = cv2.cvtColor(img1, cv2.COLOR_RGB2Luv)
    l,u,v1 = cv2.split(luv)   
    a4 = n-o+v1+h
    t4,a4 = cv2.threshold(a4,190,255,cv2.THRESH_BINARY)
    a4 = cv2.morphologyEx(a4, cv2.MORPH_OPEN, kernel)
    a4 = cv2.erode(a4,kernel,iterations = 1)
    a4 = cv2.dilate(a4,kernel,iterations = 5)
    
    #merge
    img_new = cv2.merge((a3, a2*255, a4))
    cv2.imwrite('/home/joe/Desktop/sgan/train/seg/seg%s.jpg' %k, img_new)
    k = k+1
            

print ("********************train data complete****************")
###############################################################################