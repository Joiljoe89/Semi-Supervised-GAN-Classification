#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:23:22 2019

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
# training images
dirName = '/home/joe/Desktop/sgan/train/clustering/';
# Get the list of all files in directory tree at given path
listOfFiles = getListOfFiles(dirName)
listOfFiles = sorted(listOfFiles)

k = 0
w1 = 200
h1 = 200

# Print the files
for elem in listOfFiles:
    g = 0
    x = 0
    y = 0
    print(elem[40::])
    # Load an color image in colour
    img = cv2.imread('/home/joe/Desktop/sgan/train/clustering/'+elem[40::])
    # Convert RGB to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Convert RGB to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    # Convert RGB to XYZ
    xyz = cv2.cvtColor(img, cv2.COLOR_RGB2XYZ)
    
    # Convert RGB to lab
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    
    # Convert RGB to YCrCb
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    
    
    k = k+1
    #patches
    for i in range(0,16):
        for j in range(0,12):
            cropped_rgb = img[y:y+h1, x:x+w1].copy()
            y = y+h1
            cv2.imwrite('/home/joe/Desktop/sgan/train/patch/rgb/rgb%s%s.jpg' %(k-1,g+1), cropped_rgb)
            
            g = g+1
        x = x+w1
        y = 0
            

print ("********************train data complete****************")
###############################################################################