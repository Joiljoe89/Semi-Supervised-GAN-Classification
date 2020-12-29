#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 10:27:37 2019

@author: joe
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

from mpl_toolkits import mplot3d
ax = plt.axes(projection='3d')

img = cv2.imread('/home/joe/Desktop/sgan/train/clustering/kmeans0.jpg')
#plt.hist(img.ravel(),256,[0,256]); plt.show()

Z = img.reshape((-1,3))
Z = Z.transpose()

# convert to np.float32
Z = np.float32(Z)

ax.scatter3D(Z[0], Z[1], Z[2], cmap='Greens')

plt.hist(img.ravel(),256,[0,256]); plt.show()