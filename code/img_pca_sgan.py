#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 19:57:11 2019

@author: joe
"""

import cv2
import os
import numpy as np

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

#img_weed31 = cv2.imread('/home/joe/Desktop/sgan/train/patch/train_data/weed3/ocn12.jpg')
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
#dirName_weed3 = '/home/joe/Desktop/sgan/train/patch/train_data/weed3/';
#listOfFiles_weed3 = getListOfFiles(dirName_weed3)
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
'''
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
'''
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
#l_weed3 = len(listOfFiles_weed3)+1
l_pumkin = len(listOfFiles_pumkin)+1

w, h = 200, 200
#rgb
input_disease = np.reshape(input_disease, (l_disease, w,h, 3))
input_potato = np.reshape(input_potato, (l_potato, w,h, 3))
input_grass = np.reshape(input_grass, (l_grass, w,h, 3))
#input_weed3 = np.reshape(input_weed3, (l_weed3, w,h, 3))
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
#input_weed3 = input_weed3.astype('float32') / max_value
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

print('disease',input_disease.shape,'potato',input_potato.shape,'grass',input_grass.shape,'pumkin',input_pumkin.shape)
##############################################################################
label_potato = np.zeros((l_potato,1))
label_disease = np.ones((l_disease,1))
label_grass = 2*np.ones((l_grass,1))
#label_weed3 = 3*np.ones((l_weed3,1))
label_pumkin = 4*np.ones((l_pumkin,1))
label = np.append(label_potato,label_disease,axis=0)
label = np.append(label,label_grass,axis=0)
#label = np.append(label,label_weed3,axis=0)
label = np.append(label,label_pumkin,axis=0)
##############################################################################
#input_disease_flat = input_disease.reshape(-1,l_disease)
#input_leaf_flat = input_leaf.reshape(-1,l_leaf)
#input_soil_flat = input_soil.reshape(-1,l_soil)
#input_disease_flat = input_disease.reshape(-1,l_bg)

input_train = np.append(input_potato,input_disease,axis=0)
input_train = np.append(input_train,input_grass,axis=0)
#input_train = np.append(input_train,input_weed3,axis=0)
input_train = np.append(input_train,input_pumkin,axis=0)
input_train_flat = input_train.reshape(input_train.shape[0],w*h*3)
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
feat_cols = ['pixel'+str(i) for i in range(input_train_flat.shape[1])]

import pandas as pd
df_potato_rgb = pd.DataFrame(input_train_flat,columns=feat_cols)
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
from sklearn.decomposition import PCA
pca_potato = PCA(n_components=3)
principalComponents_potato_rgb = pca_potato.fit_transform(df_potato_rgb.iloc[0::,:-1])
principal_potato_Df_rgb = pd.DataFrame(data = principalComponents_potato_rgb
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
principal_potato_Df_rgb['label'] = label[0::]
principal_potato_Df_rgb.head(10)
print('Explained variation per principal component: {}'.format(pca_potato.explained_variance_ratio_))
'''
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(df_potato_rgb.iloc[0:3000,:-1])
X_embedded.shape
principal_potato_Df_rgb1 = pd.DataFrame(data = X_embedded
             , columns = ['principal component 1', 'principal component 2'])
principal_potato_Df_rgb1['label'] = label[0:3000]
'''
#principalComponents_potato_hsl = pca_potato.fit_transform(df_potato_hsl.iloc[0::,:-1])
#principal_potato_Df_hsl = pd.DataFrame(data = principalComponents_potato_hsl
#             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
#principal_potato_Df_hsl['label'] = label[0::]
#principal_potato_Df_hsl.head(10)
#print('Explained variation per principal component: {}'.format(pca_potato.explained_variance_ratio_))
#
#principalComponents_potato_hsv = pca_potato.fit_transform(df_potato_hsv.iloc[0::,:-1])
#principal_potato_Df_hsv = pd.DataFrame(data = principalComponents_potato_hsv
#             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
#principal_potato_Df_hsv['label'] = label[0::]
#principal_potato_Df_hsv.head(10)
#print('Explained variation per principal component: {}'.format(pca_potato.explained_variance_ratio_))
#
#principalComponents_potato_lab = pca_potato.fit_transform(df_potato_lab.iloc[0::,:-1])
#principal_potato_Df_lab = pd.DataFrame(data = principalComponents_potato_lab
#             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
#principal_potato_Df_lab['label'] = label[0::]
#principal_potato_Df_lab.head(10)
#print('Explained variation per principal component: {}'.format(pca_potato.explained_variance_ratio_))
#
#principalComponents_potato_xyz = pca_potato.fit_transform(df_potato_xyz.iloc[0::,:-1])
#principal_potato_Df_xyz = pd.DataFrame(data = principalComponents_potato_xyz
#             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
#principal_potato_Df_xyz['label'] = label[0::]
#principal_potato_Df_xyz.head(10)
#print('Explained variation per principal component: {}'.format(pca_potato.explained_variance_ratio_))
#
#principalComponents_potato_ycrcb = pca_potato.fit_transform(df_potato_ycrcb.iloc[0::,:-1])
#principal_potato_Df_ycrcb = pd.DataFrame(data = principalComponents_potato_ycrcb
#             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
#principal_potato_Df_ycrcb['label'] = label[0::]
#principal_potato_Df_ycrcb.head(10)
#print('Explained variation per principal component: {}'.format(pca_potato.explained_variance_ratio_))
###############################################################################
import matplotlib.pyplot as plt
import seaborn as sns
#plt.figure(figsize=(16,16))
g = sns.pairplot(principal_potato_Df_rgb, hue="label", diag_kind="auto", palette=sns.color_palette("hls", 4),
                 vars=["principal component 1", "principal component 2", "principal component 3"])

g = sns.PairGrid(principal_potato_Df_rgb, hue="label", palette=sns.color_palette("hls", 4),
                 vars=["principal component 1", "principal component 2", "principal component 3"], diag_sharey=False)
g.map_lower(sns.kdeplot, shade=True, shade_lowest=False)
g.map_upper(sns.scatterplot,palette=sns.color_palette("hls", 4))
g.map_diag(sns.kdeplot, lw=3)
plt.savefig('/home/joe/Desktop/img_pca3_rgb_sgan.png')

plt.figure(figsize=(8,5))
sns.scatterplot(
    x="principal component 1", y="principal component 2",
    hue="label",
    palette=sns.color_palette("hls", 4),
    data=principal_potato_Df_rgb,
    legend="full",
    alpha=0.3
)
plt.savefig('/home/joe/Desktop/img_pca2_rgb2_sgan.png')


#g = sns.PairGrid(principal_potato_Df_hsl, hue="label", palette=sns.color_palette("hls", 4),
#                 vars=["principal component 1", "principal component 2", "principal component 3"], diag_sharey=False)
#g.map_lower(sns.kdeplot, shade=True, shade_lowest=False)
#g.map_upper(sns.scatterplot,palette=sns.color_palette("hls", 4))
#g.map_diag(sns.kdeplot, lw=3)
#plt.savefig('/home/joe/Desktop/img_pca/patch/rgb/img_pca_hsl3.png')
#
#plt.figure(figsize=(8,5))
#sns.scatterplot(
#    x="principal component 1", y="principal component 2", z="principal component 3",
#    hue="label",
#    palette=sns.color_palette("hls", 4),
#    data=principal_potato_Df_hsl,
#    legend="full",
#    alpha=0.3
#)
#plt.savefig('/home/joe/Desktop/img_pca/patch/rgb/img_pca_hsl.png')
#
#
#g = sns.PairGrid(principal_potato_Df_hsv, hue="label", palette=sns.color_palette("hls", 4),
#                 vars=["principal component 1", "principal component 2", "principal component 3"], diag_sharey=False)
#g.map_lower(sns.kdeplot, shade=True, shade_lowest=False)
#g.map_upper(sns.scatterplot,palette=sns.color_palette("hls", 3))
#g.map_diag(sns.kdeplot, lw=4)
#plt.savefig('/home/joe/Desktop/img_pca/patch/rgb/img_pca_hsv3.png')
#
#plt.figure(figsize=(8,5))
#sns.scatterplot(
#    x="principal component 1", y="principal component 2", z="principal component 3",
#    hue="label",
#    palette=sns.color_palette("hls", 4),
#    data=principal_potato_Df_hsv,
#    legend="full",
#    alpha=0.3
#)
#plt.savefig('/home/joe/Desktop/img_pca/patch/rgb/img_pca_hsv.png')
#
#
#
#g = sns.PairGrid(principal_potato_Df_lab, hue="label", palette=sns.color_palette("hls", 4),
#                 vars=["principal component 1", "principal component 2", "principal component 3"], diag_sharey=False)
#g.map_lower(sns.kdeplot, shade=True, shade_lowest=False)
#g.map_upper(sns.scatterplot,palette=sns.color_palette("hls", 4))
#g.map_diag(sns.kdeplot, lw=3)
#plt.savefig('/home/joe/Desktop/img_pca/patch/rgb/img_pca_lab3.png')
#
#plt.figure(figsize=(8,5))
#sns.scatterplot(
#    x="principal component 1", y="principal component 2", z="principal component 3",
#    hue="label",
#    palette=sns.color_palette("hls", 4),
#    data=principal_potato_Df_lab,
#    legend="full",
#    alpha=0.3
#)
#plt.savefig('/home/joe/Desktop/img_pca/patch/rgb/img_pca_lab.png')
#
#
#
#g = sns.PairGrid(principal_potato_Df_xyz, hue="label", palette=sns.color_palette("hls", 4),
#                 vars=["principal component 1", "principal component 2", "principal component 3"], diag_sharey=False)
#g.map_lower(sns.kdeplot, shade=True, shade_lowest=False)
#g.map_upper(sns.scatterplot,palette=sns.color_palette("hls", 4))
#g.map_diag(sns.kdeplot, lw=3)
#plt.savefig('/home/joe/Desktop/img_pca/patch/rgb/img_pca_xyz3.png')
#
#plt.figure(figsize=(8,5))
#sns.scatterplot(
#    x="principal component 1", y="principal component 2", z="principal component 3",
#    hue="label",
#    palette=sns.color_palette("hls", 4),
#    data=principal_potato_Df_xyz,
#    legend="full",
#    alpha=0.3
#)
#plt.savefig('/home/joe/Desktop/img_pca/patch/rgb/img_pca_xyz.png')
#
#
#g = sns.PairGrid(principal_potato_Df_ycrcb, hue="label", palette=sns.color_palette("hls", 4),
#                 vars=["principal component 1", "principal component 2", "principal component 3"], diag_sharey=False)
#g.map_lower(sns.kdeplot, shade=True, shade_lowest=False)
#g.map_upper(sns.scatterplot,palette=sns.color_palette("hls", 4))
#g.map_diag(sns.kdeplot, lw=3)
#plt.savefig('/home/joe/Desktop/img_pca/patch/rgb/img_pca_ycrcb3.png')
#
#
#plt.figure(figsize=(8,5))
#sns.scatterplot(
#    x="principal component 1", y="principal component 2", z="principal component 3",
#    hue="label",
#    palette=sns.color_palette("hls", 4),
#    data=principal_potato_Df_ycrcb,
#    legend="full",
#    alpha=0.3
#)
#plt.savefig('/home/joe/Desktop/img_pca/patch/rgb/img_pca_ycrcb.png')
###############################################################################
'''
g1 = (principal_potato_Df['principal component 1'][0:10],principal_potato_Df['principal component 2'][0:10])
g2 = (principal_potato_Df['principal component 1'][900:910],principal_potato_Df['principal component 2'][900:910])
g3 = (principal_potato_Df['principal component 1'][3100:3110],principal_potato_Df['principal component 2'][3100:3110])
data = (g1, g2, g3)
colors = ("red", "green", "blue")
groups = ("disease", "leaf", "soil")

# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, axisbg="1.0")

for data, color, group in zip(data, colors, groups):
    x, y = data
    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

plt.title('Matplot scatter plot')
plt.legend(loc=2)
plt.show()
#########################
colors = (0,0,0)
area = np.pi*3

# Plot
plt.scatter(principal_potato_Df['principal component 1'][0:1000], principal_potato_Df['principal component 2'][0:1000], s=area, c=colors, alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
'''
###############################################################################
