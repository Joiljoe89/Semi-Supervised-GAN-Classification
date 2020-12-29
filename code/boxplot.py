#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:45:40 2019

@author: joe
"""

import cv2
import os
import numpy as np
from matplotlib import pyplot

##############################################################################
    #For the given path, get the List of all files in the directory tree

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
img_potato1 = cv2.imread('/home/joe/Desktop/sgan/train/patch/train_data/potato/ocn0.jpg')
img_disease1 = cv2.imread('/home/joe/Desktop/sgan/train/patch/train_data/disease/rgb025.jpg')

#hsl_disease1 = cv2.cvtColor(img_disease1, cv2.COLOR_RGB2HLS)
#hsv_disease1 = cv2.cvtColor(img_disease1, cv2.COLOR_RGB2HSV)
#lab_disease1 = cv2.cvtColor(img_disease1, cv2.COLOR_RGB2LAB)
#xyz_disease1 = cv2.cvtColor(img_disease1, cv2.COLOR_RGB2XYZ)
#ycrcb_disease1 = cv2.cvtColor(img_disease1, cv2.COLOR_RGB2YCrCb)

img_grass1 = cv2.imread('/home/joe/Desktop/sgan/train/patch/train_data/grass/ocn10.jpg')
#hsl_leaf1 = cv2.cvtColor(img_disease1, cv2.COLOR_RGB2HLS)
#hsv_leaf1 = cv2.cvtColor(img_disease1, cv2.COLOR_RGB2HSV)
#lab_leaf1 = cv2.cvtColor(img_disease1, cv2.COLOR_RGB2LAB)
#xyz_leaf1 = cv2.cvtColor(img_disease1, cv2.COLOR_RGB2XYZ)
#ycrcb_leaf1 = cv2.cvtColor(img_disease1, cv2.COLOR_RGB2YCrCb)

img_weed31 = cv2.imread('/home/joe/Desktop/sgan/train/patch/train_data/weed3/ocn12.jpg')
#hsl_soil1 = cv2.cvtColor(img_disease1, cv2.COLOR_RGB2HLS)
#hsv_soil1 = cv2.cvtColor(img_disease1, cv2.COLOR_RGB2HSV)
#lab_soil1 = cv2.cvtColor(img_disease1, cv2.COLOR_RGB2LAB)
#xyz_soil1 = cv2.cvtColor(img_disease1, cv2.COLOR_RGB2XYZ)
#ycrcb_soil1 = cv2.cvtColor(img_disease1, cv2.COLOR_RGB2YCrCb)

img_pumkin1 = cv2.imread('/home/joe/Desktop/sgan/train/patch/train_data/pumkin/ocn2.jpg')
#hsl_bg1 = cv2.cvtColor(img_disease1, cv2.COLOR_RGB2HLS)
#hsv_bg1 = cv2.cvtColor(img_disease1, cv2.COLOR_RGB2HSV)
#lab_bg1 = cv2.cvtColor(img_disease1, cv2.COLOR_RGB2LAB)
#xyz_bg1 = cv2.cvtColor(img_disease1, cv2.COLOR_RGB2XYZ)
#ycrcb_bg1 = cv2.cvtColor(img_disease1, cv2.COLOR_RGB2YCrCb)


dirName_potato = '/home/joe/Desktop/sgan/train/patch/train_data/potato/';
listOfFiles_potato = getListOfFiles(dirName_potato)
dirName_disease = '/home/joe/Desktop/sgan/train/patch/train_data/disease/';
listOfFiles_disease = getListOfFiles(dirName_disease)
dirName_grass = '/home/joe/Desktop/sgan/train/patch/train_data/grass/';
listOfFiles_grass = getListOfFiles(dirName_grass)
dirName_weed3 = '/home/joe/Desktop/sgan/train/patch/train_data/weed3/';
listOfFiles_weed3 = getListOfFiles(dirName_weed3)
dirName_pumkin = '/home/joe/Desktop/sgan/train/patch/train_data/pumkin/';
listOfFiles_pumkin = getListOfFiles(dirName_pumkin)

# Print the files
for elem_disease in listOfFiles_disease:
    #print(elem_in[53::])
    img_disease = cv2.imread('/home/joe/Desktop/sgan/train/patch/train_data/disease/'+elem_disease[53::])
#    hsl_disease = cv2.cvtColor(img_disease, cv2.COLOR_RGB2HLS)
#    hsv_disease = cv2.cvtColor(img_disease, cv2.COLOR_RGB2HSV)
#    lab_disease = cv2.cvtColor(img_disease, cv2.COLOR_RGB2LAB)
#    xyz_disease = cv2.cvtColor(img_disease, cv2.COLOR_RGB2XYZ)
#    ycrcb_disease = cv2.cvtColor(img_disease, cv2.COLOR_RGB2YCrCb)
    
    input_disease = np.append(img_disease1,img_disease, axis=0)
    img_disease1 = input_disease
#    hsl_input_disease = np.append(hsl_disease1,hsl_disease, axis=0)
#    hsl_disease1 = hsl_input_disease
#    hsv_input_disease = np.append(hsv_disease1,hsv_disease, axis=0)
#    hsv_disease1 = hsv_input_disease
#    lab_input_disease = np.append(lab_disease1,lab_disease, axis=0)
#    lab_disease1 = lab_input_disease
#    xyz_input_disease = np.append(xyz_disease1,xyz_disease, axis=0)
#    xyz_disease1 = xyz_input_disease
#    ycrcb_input_disease = np.append(ycrcb_disease1,ycrcb_disease, axis=0)
#    ycrcb_disease1 = ycrcb_input_disease
print('disease',img_disease1.shape)

# Print the files
for elem_potato in listOfFiles_potato:
    #print(elem_in[53::])
    img_potato = cv2.imread('/home/joe/Desktop/sgan/train/patch/train_data/potato/'+elem_potato[52::])
#    hsl_bg = cv2.cvtColor(img_bg, cv2.COLOR_RGB2HLS)
#    hsv_bg = cv2.cvtColor(img_bg, cv2.COLOR_RGB2HSV)
#    lab_bg = cv2.cvtColor(img_bg, cv2.COLOR_RGB2LAB)
#    xyz_bg = cv2.cvtColor(img_bg, cv2.COLOR_RGB2XYZ)
#    ycrcb_bg = cv2.cvtColor(img_bg, cv2.COLOR_RGB2YCrCb)
    
    input_potato = np.append(img_potato1,img_potato, axis=0)
    img_potato1 = input_potato
#    hsl_input_bg = np.append(hsl_bg1,hsl_bg, axis=0)
#    hsl_bg1 = hsl_input_bg
#    hsv_input_bg = np.append(hsv_bg1,hsv_bg, axis=0)
#    hsv_bg1 = hsv_input_bg
#    lab_input_bg = np.append(lab_bg1,lab_bg, axis=0)
#    lab_bg1 = lab_input_bg
#    xyz_input_bg = np.append(xyz_bg1,xyz_bg, axis=0)
#    xyz_bg1 = xyz_input_bg
#    ycrcb_input_bg = np.append(ycrcb_bg1,ycrcb_bg, axis=0)
#    ycrcb_bg1 = ycrcb_input_bg
print('potato',img_potato1.shape)

# Print the files
for elem_grass in listOfFiles_grass:
    #print(elem_in[53::])
    img_grass = cv2.imread('/home/joe/Desktop/sgan/train/patch/train_data/grass/'+elem_grass[51::])
#    hsl_leaf = cv2.cvtColor(img_leaf, cv2.COLOR_RGB2HLS)
#    hsv_leaf = cv2.cvtColor(img_leaf, cv2.COLOR_RGB2HSV)
#    lab_leaf = cv2.cvtColor(img_leaf, cv2.COLOR_RGB2LAB)
#    xyz_leaf = cv2.cvtColor(img_leaf, cv2.COLOR_RGB2XYZ)
#    ycrcb_leaf = cv2.cvtColor(img_leaf, cv2.COLOR_RGB2YCrCb)
    
    input_grass = np.append(img_grass1,img_grass, axis=0)
    img_grass1 = input_grass
#    hsl_input_leaf = np.append(hsl_leaf1,hsl_leaf, axis=0)
#    hsl_leaf1 = hsl_input_leaf
#    hsv_input_leaf = np.append(hsv_leaf1,hsv_leaf, axis=0)
#    hsv_leaf1 = hsv_input_leaf
#    lab_input_leaf = np.append(lab_leaf1,lab_leaf, axis=0)
#    lab_leaf1 = lab_input_leaf
#    xyz_input_leaf = np.append(xyz_leaf1,xyz_leaf, axis=0)
#    xyz_leaf1 = xyz_input_leaf
#    ycrcb_input_leaf = np.append(ycrcb_leaf1,ycrcb_leaf, axis=0)
#    ycrcb_leaf1 = ycrcb_input_leaf
print('grass',img_grass1.shape)

# Print the files
for elem_weed3 in listOfFiles_weed3:
    #print(elem_in[53::])
    img_weed3 = cv2.imread('/home/joe/Desktop/sgan/train/patch/train_data/weed3/'+elem_weed3[51::])
#    hsl_soil = cv2.cvtColor(img_soil, cv2.COLOR_RGB2HLS)
#    hsv_soil = cv2.cvtColor(img_soil, cv2.COLOR_RGB2HSV)
#    lab_soil = cv2.cvtColor(img_soil, cv2.COLOR_RGB2LAB)
#    xyz_soil = cv2.cvtColor(img_soil, cv2.COLOR_RGB2XYZ)
#    ycrcb_soil = cv2.cvtColor(img_soil, cv2.COLOR_RGB2YCrCb)
    
    input_weed3 = np.append(img_weed31,img_weed3, axis=0)
    img_weed31 = input_weed3
#    hsl_input_soil = np.append(hsl_soil1,hsl_soil, axis=0)
#    hsl_soil1 = hsl_input_soil
#    hsv_input_soil = np.append(hsv_soil1,hsv_soil, axis=0)
#    hsv_soil1 = hsv_input_soil
#    lab_input_soil = np.append(lab_soil1,lab_soil, axis=0)
#    lab_soil1 = lab_input_soil
#    xyz_input_soil = np.append(xyz_soil1,xyz_soil, axis=0)
#    xyz_soil1 = xyz_input_soil
#    ycrcb_input_soil = np.append(ycrcb_soil1,ycrcb_soil, axis=0)
#    ycrcb_soil1 = ycrcb_input_soil
print('weed3',img_weed31.shape)

for elem_pumkin in listOfFiles_pumkin:
    #print(elem_in[53::])
    img_pumkin = cv2.imread('/home/joe/Desktop/sgan/train/patch/train_data/pumkin/'+elem_pumkin[52::])
#    hsl_soil = cv2.cvtColor(img_soil, cv2.COLOR_RGB2HLS)
#    hsv_soil = cv2.cvtColor(img_soil, cv2.COLOR_RGB2HSV)
#    lab_soil = cv2.cvtColor(img_soil, cv2.COLOR_RGB2LAB)
#    xyz_soil = cv2.cvtColor(img_soil, cv2.COLOR_RGB2XYZ)
#    ycrcb_soil = cv2.cvtColor(img_soil, cv2.COLOR_RGB2YCrCb)
    
    input_pumkin = np.append(img_pumkin1,img_pumkin, axis=0)
    img_pumkin1 = input_pumkin
#    hsl_input_soil = np.append(hsl_soil1,hsl_soil, axis=0)
#    hsl_soil1 = hsl_input_soil
#    hsv_input_soil = np.append(hsv_soil1,hsv_soil, axis=0)
#    hsv_soil1 = hsv_input_soil
#    lab_input_soil = np.append(lab_soil1,lab_soil, axis=0)
#    lab_soil1 = lab_input_soil
#    xyz_input_soil = np.append(xyz_soil1,xyz_soil, axis=0)
#    xyz_soil1 = xyz_input_soil
#    ycrcb_input_soil = np.append(ycrcb_soil1,ycrcb_soil, axis=0)
#    ycrcb_soil1 = ycrcb_input_soil
print('pumkin',img_pumkin1.shape)

print ("********************train data loading complete****************")
###############################################################################
l_disease = len(listOfFiles_disease)+1
l_potato = len(listOfFiles_potato)+1
l_grass = len(listOfFiles_grass)+1
l_weed3 = len(listOfFiles_weed3)+1
l_pumkin = len(listOfFiles_pumkin)+1

w, h = 200, 200
#rgb
input_disease = np.reshape(input_disease, (l_disease, w,h, 3))
input_potato = np.reshape(input_potato, (l_potato, w,h, 3))
input_grass = np.reshape(input_grass, (l_grass, w,h, 3))
input_weed3 = np.reshape(input_weed3, (l_weed3, w,h, 3))
input_pumkin = np.reshape(input_pumkin, (l_pumkin, w,h, 3))
##hsl
#hsl_input_disease = np.reshape(hsl_input_disease, (l_disease, w,h, 3))
#hsl_input_bg = np.reshape(hsl_input_bg, (l_bg, w,h, 3))
#hsl_input_leaf = np.reshape(hsl_input_leaf, (l_leaf, w,h, 3))
#hsl_input_soil = np.reshape(hsl_input_soil, (l_soil, w,h, 3))
##hsv
#hsv_input_disease = np.reshape(hsv_input_disease, (l_disease, w,h, 3))
#hsv_input_bg = np.reshape(hsv_input_bg, (l_bg, w,h, 3))
#hsv_input_leaf = np.reshape(hsv_input_leaf, (l_leaf, w,h, 3))
#hsv_input_soil = np.reshape(hsv_input_soil, (l_soil, w,h, 3))
##lab
#lab_input_disease = np.reshape(lab_input_disease, (l_disease, w,h, 3))
#lab_input_bg = np.reshape(lab_input_bg, (l_bg, w,h, 3))
#lab_input_leaf = np.reshape(lab_input_leaf, (l_leaf, w,h, 3))
#lab_input_soil = np.reshape(lab_input_soil, (l_soil, w,h, 3))
##xyz
#xyz_input_disease = np.reshape(xyz_input_disease, (l_disease, w,h, 3))
#xyz_input_bg = np.reshape(xyz_input_bg, (l_bg, w,h, 3))
#xyz_input_leaf = np.reshape(xyz_input_leaf, (l_leaf, w,h, 3))
#xyz_input_soil = np.reshape(xyz_input_soil, (l_soil, w,h, 3))
##ycrcb
#ycrcb_input_disease = np.reshape(ycrcb_input_disease, (l_disease, w,h, 3))
#ycrcb_input_bg = np.reshape(ycrcb_input_bg, (l_bg, w,h, 3))
#ycrcb_input_leaf = np.reshape(ycrcb_input_leaf, (l_leaf, w,h, 3))
#ycrcb_input_soil = np.reshape(ycrcb_input_soil, (l_soil, w,h, 3))

max_value = 255.0
input_disease = input_disease.astype('float32') / max_value
input_potato = input_potato.astype('float32') / max_value
input_grass = input_grass.astype('float32') / max_value
input_weed3 = input_weed3.astype('float32') / max_value
input_pumkin = input_pumkin.astype('float32') / max_value

#hsl_input_disease = hsl_input_disease.astype('float32') / max_value
#hsl_input_bg = hsl_input_bg.astype('float32') / max_value
#hsl_input_leaf = hsl_input_leaf.astype('float32') / max_value
#hsl_input_soil = hsl_input_soil.astype('float32') / max_value
#
#hsv_input_disease = hsv_input_disease.astype('float32') / max_value
#hsv_input_bg = hsv_input_bg.astype('float32') / max_value
#hsv_input_leaf = hsv_input_leaf.astype('float32') / max_value
#hsv_input_soil = hsv_input_soil.astype('float32') / max_value
#
#lab_input_disease = lab_input_disease.astype('float32') / max_value
#lab_input_bg = lab_input_bg.astype('float32') / max_value
#lab_input_leaf = lab_input_leaf.astype('float32') / max_value
#lab_input_soil = lab_input_soil.astype('float32') / max_value
#
#xyz_input_disease = xyz_input_disease.astype('float32') / max_value
#xyz_input_bg = xyz_input_bg.astype('float32') / max_value
#xyz_input_leaf = xyz_input_leaf.astype('float32') / max_value
#xyz_input_soil = xyz_input_soil.astype('float32') / max_value
#
#ycrcb_input_disease = ycrcb_input_disease.astype('float32') / max_value
#ycrcb_input_bg = ycrcb_input_bg.astype('float32') / max_value
#ycrcb_input_leaf = ycrcb_input_leaf.astype('float32') / max_value
#ycrcb_input_soil = ycrcb_input_soil.astype('float32') / max_value

print('disease',input_disease.shape,'potato',input_potato.shape,'grass',input_grass.shape,'weed3',input_weed3.shape,'pumkin',input_pumkin.shape)
##############################################################################
label_potato = np.zeros((l_potato,1))
label_disease = np.ones((l_disease,1))
label_grass = 2*np.ones((l_grass,1))
label_weed3 = 3*np.ones((l_weed3,1))
label_pumkin = 4*np.ones((l_pumkin,1))
label = np.append(label_potato,label_disease,axis=0)
label = np.append(label,label_grass,axis=0)
label = np.append(label,label_weed3,axis=0)
label = np.append(label,label_pumkin,axis=0)
##############################################################################
#input_disease_flat = input_disease.reshape(-1,l_disease)
#input_leaf_flat = input_leaf.reshape(-1,l_leaf)
#input_soil_flat = input_soil.reshape(-1,l_soil)
#input_disease_flat = input_disease.reshape(-1,l_bg)

input_train = np.append(input_potato,input_disease,axis=0)
input_train = np.append(input_train,input_grass,axis=0)
input_train = np.append(input_train,input_weed3,axis=0)
input_train = np.append(input_train,input_pumkin,axis=0)
input_train_flat = input_train.reshape(input_train.shape[0]*w*h,3)
input_train_flat1 = input_train_flat.transpose()
input_train_flat2 = input_train_flat1.reshape(input_train.shape[0]*w*h,3)
#input_train_flat = input_train.reshape(-1,input_train.shape[0])
#input_train_flat = np.transpose(input_train_flat)
#hsl_input_train = np.append(hsl_input_disease,input_leaf,axis=0)
#hsl_input_train = np.append(hsl_input_train,input_soil,axis=0)
#hsl_input_train = np.append(hsl_input_train,input_bg,axis=0)
#hsl_input_train_flat = hsl_input_train.reshape(hsl_input_train.shape[0],w*h*3)
#
#hsv_input_train = np.append(hsv_input_disease,input_leaf,axis=0)
#hsv_input_train = np.append(hsv_input_train,input_soil,axis=0)
#hsv_input_train = np.append(hsv_input_train,input_bg,axis=0)
#hsv_input_train_flat = hsv_input_train.reshape(hsv_input_train.shape[0],w*h*3)
#
#lab_input_train = np.append(lab_input_disease,input_leaf,axis=0)
#lab_input_train = np.append(lab_input_train,input_soil,axis=0)
#lab_input_train = np.append(lab_input_train,input_bg,axis=0)
#lab_input_train_flat = lab_input_train.reshape(lab_input_train.shape[0],w*h*3)
#
#xyz_input_train = np.append(xyz_input_disease,input_leaf,axis=0)
#xyz_input_train = np.append(xyz_input_train,input_soil,axis=0)
#xyz_input_train = np.append(xyz_input_train,input_bg,axis=0)
#xyz_input_train_flat = xyz_input_train.reshape(xyz_input_train.shape[0],w*h*3)
#
#ycrcb_input_train = np.append(ycrcb_input_disease,input_leaf,axis=0)
#ycrcb_input_train = np.append(ycrcb_input_train,input_soil,axis=0)
#ycrcb_input_train = np.append(ycrcb_input_train,input_bg,axis=0)
#ycrcb_input_train_flat = ycrcb_input_train.reshape(ycrcb_input_train.shape[0],w*h*3)

##############################################################################
feat_cols = ['pixel'+str(i) for i in range(input_train_flat1.shape[1])]

import pandas as pd
df_potato_rgb = pd.DataFrame(input_train_flat1,columns=feat_cols)
#df_potato_hsl = pd.DataFrame(hsl_input_train_flat,columns=feat_cols)
#df_potato_hsv = pd.DataFrame(hsv_input_train_flat,columns=feat_cols)
#df_potato_lab = pd.DataFrame(lab_input_train_flat,columns=feat_cols)
#df_potato_xyz = pd.DataFrame(xyz_input_train_flat,columns=feat_cols)
#df_potato_ycrcb = pd.DataFrame(ycrcb_input_train_flat,columns=feat_cols)

#df_potato = df_potato.transpose()
df_potato_rgb['label'] = label
#df_potato_hsl['label'] = label
#df_potato_hsv['label'] = label
#df_potato_lab['label'] = label
#df_potato_xyz['label'] = label
#df_potato_ycrcb['label'] = label
print('Size of the dataframe: {}'.format(df_potato_rgb.shape))
df_potato_rgb.head()

##############################################################################
boxplot = df_potato_rgb.boxplot(column=['Col1', 'Col2', 'Col3'])
##############################################################################

'''
input_ocn = np.reshape(input_ocn, (71, 3000, 4000, 3))
input_seg = np.reshape(input_seg, (71, 3000, 4000, 3))
x_train_ocn = input_ocn[0:57,:,:,:]
x_test_ocn = input_ocn[57:,:,:,:]
x_train_ocn = x_train_ocn.astype('float32') / 255.
x_test_ocn = x_test_ocn.astype('float32') / 255.

x_train_seg = input_seg[0:57,:,:,:]
x_test_seg = input_seg[57:,:,:,:]
x_train_seg = x_train_seg.astype('float32') / 255.
x_test_seg = x_test_seg.astype('float32') / 255.

#plt.hist(img.ravel(),256,[0,256]); plt.show() #histogram

# Conv Autoencoder

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

input_img = Input(shape=(3000, 4000, 3))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')



from keras.callbacks import TensorBoard

autoencoder.fit(x_train_ocn, x_train_seg,
                epochs=3,
                batch_size=1,
                shuffle=True,
                validation_data=(x_test_ocn, x_test_seg),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

autoencoder.save('AE_model_1.h5')  # creates a HDF5 file 'my_model.h5'


from keras.models import load_model
model = load_model('AE_model.h5')
model.summary()

#https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
from matplotlib import pyplot
# summarize filter shapes
for layer in model.layers:
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# get filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)

# retrieve weights from the second hidden layer
filters, biases = model.layers[3].get_weights()
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	# plot each channel separately
	for j in range(3):
		# specify subplot and turn of axis
		ax = pyplot.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(f[:, :, j], cmap='gray')
		ix += 1
# show the figure
pyplot.show()

model = autoencoder
# plot feature map of first conv layer for given image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
# load the model
#model = VGG16()
# redefine model to output right after the first hidden layer
model = Model(inputs=model.inputs, outputs=model.layers[5].output)
model.summary()
# load the image with the required shape
#img = load_img('bird.jpg', target_size=(224, 224))
img = load_img('/home/joe/Desktop/sgan/train/clustering/kmeans0.jpg')
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
#img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model.predict(img)
# plot all 64 maps in an 8x8 squares
square = 1
ix = 1
for _ in range(square):
	for _ in range(square):
		# specify subplot and turn of axis
		ax = pyplot.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(feature_maps[0, :, :, ix-4], cmap='gray')
		ix += 1
# show the figure
pyplot.savefig("/home/joe/Desktop/feature2.png")
pyplot.show()

# visualize feature maps output from each block in the vgg model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
# load the model
model = VGG16()
# redefine model to output right after the first hidden layer
ixs = [2, 5, 9, 13, 17]
outputs = [model.layers[i].output for i in ixs]
model = Model(inputs=model.inputs, outputs=outputs)
# load the image with the required shape
img = load_img('bird.jpg', target_size=(224, 224))
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model.predict(img)
# plot the output from each block
square = 8
for fmap in feature_maps:
	# plot all 64 maps in an 8x8 squares
	ix = 1
	for _ in range(square):
		for _ in range(square):
			# specify subplot and turn of axis
			ax = pyplot.subplot(square, square, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
			ix += 1
	# show the figure
	pyplot.show()




img = load_img('/home/joe/Desktop/sgan/train/clustering/kmeans0.jpg')
x_test = np.reshape(img, (1, 3000, 4000, 3))
x_test = x_test.astype('float32') / 255.
decoded_imgs = model.predict(x_test)
print(decoded_imgs.shape)
x_test_img = np.reshape(decoded_imgs, (3000, 4000, 3))
pyplot.imshow(x_test_img)
pyplot.savefig('/home/joe/Desktop/decoded3.png')
pyplot.show()



n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(3000, 4000))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(3000, 4000))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(1, n, i)
    plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
'''