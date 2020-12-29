#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:23:52 2019

@author: joe
"""

from keras.applications.densenet import DenseNet121
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input
from keras.models import Model
import numpy as np
import cv2
import matplotlib.pyplot as plt

base_model = DenseNet121(weights='imagenet')
base_model.summary()

model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv4_block9_1_relu').output)

img_in = cv2.imread('/home/joe/Desktop/sgan/train/patch/rgb/rgb108.jpg')
img_in = cv2.resize(img_in, (224,224), interpolation = cv2.INTER_CUBIC)
x = image.img_to_array(img_in)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features1 = model.predict(x)
feature1 = features1[:,:,:,9]

print(features1.shape)
feature1 = np.reshape(feature1, (14,14))
plt.imshow(feature1)
plt.show()
