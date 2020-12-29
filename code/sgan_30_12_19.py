#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:37:17 2019

@author: joe
"""

from __future__ import print_function, division

#from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

import matplotlib.pyplot as plt
import os
import cv2

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

class SGAN:
    def __init__(self):
        self.img_rows = 200
        self.img_cols = 200
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 5
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=['binary_crossentropy', 'categorical_crossentropy'],
            loss_weights=[0.5, 0.5],
            optimizer=optimizer,
            metrics=['accuracy']
        )

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        noise = Input(shape=(100,))
        img = self.generator(noise)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid, _ = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(noise, valid)
        self.combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(512 * 14 * 14, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((14, 14, 512)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(3, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())

        model.summary()

        img = Input(shape=self.img_shape)

        features = model(img)
        valid = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes+1, activation="softmax")(features)

        return Model(img, [valid, label])

    def train(self, epochs, batch_size=128, sample_interval=50):
        '''
        # Load the dataset
        (X_train, y_train), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)
        '''
        
        # Load the dataset
       
        img_c1 = cv2.imread('/home/joe/Desktop/sgan/train/patch/train_data/potato/ocn0.jpg')
        img_c2 = cv2.imread('/home/joe/Desktop/sgan/train/patch/train_data/disease/rgb025.jpg')
        img_c3 = cv2.imread('/home/joe/Desktop/sgan/train/patch/train_data/grass/ocn10.jpg')
        img_c4 = cv2.imread('/home/joe/Desktop/sgan/train/patch/train_data/weed3/ocn12.jpg')
        img_c5 = cv2.imread('/home/joe/Desktop/sgan/train/patch/train_data/pumkin/ocn2.jpg')
        
        
        # training images
        
        dirName_c1 = '/home/joe/Desktop/sgan/train/patch/train_data/potato';
        dirName_c2 = '/home/joe/Desktop/sgan/train/patch/train_data/disease';
        dirName_c3 = '/home/joe/Desktop/sgan/train/patch/train_data/grass';
        dirName_c4 = '/home/joe/Desktop/sgan/train/patch/train_data/weed3';
        dirName_c5 = '/home/joe/Desktop/sgan/train/patch/train_data/pumkin';
        # Get the list of all files in directory tree at given path
        listOfFiles_c1 = getListOfFiles(dirName_c1)
        listOfFiles_c1 = sorted(listOfFiles_c1)
        listOfFiles_c2 = getListOfFiles(dirName_c2)
        listOfFiles_c2 = sorted(listOfFiles_c2)
        listOfFiles_c3 = getListOfFiles(dirName_c3)
        listOfFiles_c3 = sorted(listOfFiles_c3)
        listOfFiles_c4 = getListOfFiles(dirName_c4)
        listOfFiles_c4 = sorted(listOfFiles_c4)
        listOfFiles_c5 = getListOfFiles(dirName_c5)
        listOfFiles_c5 = sorted(listOfFiles_c5)
        
        # Print the files
        for elem_c1 in listOfFiles_c1:
            print(elem_c1[52::])
            img_in = cv2.imread('/home/joe/Desktop/sgan/train/patch/train_data/potato'+elem_c1[52::])
            #img_in = cv2.resize(img_in, (w,h), interpolation = cv2.INTER_CUBIC) #INTER_NEAREST,INTER_LINEAR,INTER_CUBIC,INTER_LANCZOS4 
            #print(img_in.shape)
            input_img = np.append(img_c1,img_in, axis=0)
            img_c1 = input_img
            #print(img_in1.shape)
        
        input_img = 0
        for elem_c2 in listOfFiles_c2:
            img_in = cv2.imread('/home/joe/Desktop/sgan/train/patch/train_data/disease'+elem_c2[53::])
            input_img = np.append(img_c2,img_in, axis=0)
            img_c2 = input_img
        
        input_img = 0
        for elem_c3 in listOfFiles_c3:
            img_in = cv2.imread('/home/joe/Desktop/sgan/train/patch/train_data/grass'+elem_c3[51::])
            input_img = np.append(img_c3,img_in, axis=0)
            img_c3 = input_img
        
        input_img = 0
        for elem_c4 in listOfFiles_c4:
            img_in = cv2.imread('/home/joe/Desktop/sgan/train/patch/train_data/weed3'+elem_c4[51::])
            input_img = np.append(img_c4,img_in, axis=0)
            img_c4 = input_img
        
        input_img = 0
        for elem_c5 in listOfFiles_c5:
            img_in = cv2.imread('/home/joe/Desktop/sgan/train/patch/train_data/pumkin'+elem_c5[52::])
            input_img = np.append(img_c5,img_in, axis=0)
            img_c5 = input_img
        
        input_img = 0
        print(img_c1.shape,img_c2.shape,img_c3.shape,img_c4.shape,img_c5.shape)
        nb_img_c1 = int(img_c1.shape[0]/200)
        nb_img_c2 = int(img_c2.shape[0]/200)
        nb_img_c3 = int(img_c3.shape[0]/200)
        nb_img_c4 = int(img_c4.shape[0]/200)
        nb_img_c5 = int(img_c5.shape[0]/200)
        max_value = 255.0
        img_c1= img_c1.astype('float32') / max_value
        img_c2 =img_c2.astype('float32') / max_value
        img_c3 =img_c3.astype('float32') / max_value
        img_c4 =img_c4.astype('float32') / max_value
        img_c5 =img_c5.astype('float32') / max_value
        X_train = np.append(img_c1,img_c2, axis=0)
        X_train = np.append(X_train,img_c3, axis=0)
        X_train = np.append(X_train,img_c4, axis=0)
        X_train = np.append(X_train,img_c5, axis=0)
        X_train = X_train.reshape(nb_img_c1+nb_img_c2+nb_img_c3+nb_img_c4+nb_img_c5,200,200,3)
        y_train = np.append(np.zeros(nb_img_c1),np.ones(nb_img_c2))
        y_train = np.append(y_train,2*np.ones(nb_img_c3))
        y_train = np.append(y_train,3*np.ones(nb_img_c4))
        y_train = np.append(y_train,4*np.ones(nb_img_c5))
        y_train = y_train.reshape(-1, 1)
        
        
        print ("********************train data complete****************")
        
        # Class weights:
        # To balance the difference in occurences of digit class labels.
        # 50% of labels that the discriminator trains on are 'fake'.
        # Weight = 1 / frequency
        half_batch = batch_size // 2
        cw1 = {0: 1, 1: 1}
        cw2 = {i: self.num_classes / half_batch for i in range(self.num_classes)}
        cw2[self.num_classes] = 1 / half_batch

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # One-hot encoding of labels
            labels = to_categorical(y_train[idx], num_classes=self.num_classes+1)
            fake_labels = to_categorical(np.full((batch_size, 1), self.num_classes), num_classes=self.num_classes+1)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, labels], class_weight=[cw1, cw2])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels], class_weight=[cw1, cw2])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid, class_weight=[cw1, cw2])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                self.save_model()

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "mnist_sgan_generator")
        save(self.discriminator, "mnist_sgan_discriminator")
        save(self.combined, "mnist_sgan_adversarial")


if __name__ == '__main__':
    sgan = SGAN()
    sgan.train(epochs=2000, batch_size=32, sample_interval=50)
    
    
##########################################################################
'''
from keras.models import model_from_json

# load json and create model
json_file = open('/home/joe/Desktop/sgan/sgan_test/model/mnist_sgan_discriminator.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.summary()
# load weights into new model
loaded_model.load_weights("/home/joe/Desktop/sgan/sgan_test/model/mnist_sgan_discriminator_weights.hdf5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
img_test = cv2.imread('/home/joe/Desktop/sgan/train/patch/train_data/weed/ocn0.jpg')
img_test = img_test.reshape(None,200,200,3)
score = loaded_model.evaluate(img_test)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
'''