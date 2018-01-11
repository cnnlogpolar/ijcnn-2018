#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 20:05:07 2017

@author: marta
"""

import numpy
import os
from scipy.misc import imread
from scipy.misc import imresize
import cv2

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

########################### LOADING IMAGE DATA ################################
base_dir = 'TUDarmstadt_fred/'
img_dir = 'PNGImages/'
img_ext = 'png'
lbl_dir = 'Annotations/'
lbl_ext = 'txt'
import glob

img_paths = glob.glob(base_dir + img_dir + '**/*.' + img_ext, recursive=True)
lbl_paths = glob.glob(base_dir + lbl_dir + '**/*.' + lbl_ext, recursive=True)

# Image settings
h = 150 # height
w = 200 # width
dp = 3 # color RGB
n_imgs = len(img_paths) # number of images

imgs = numpy.zeros((n_imgs,h,w,dp), dtype=numpy.float32)
lbls = ['' for i in range(n_imgs)]






def readimage(directory):
    
    img_paths = glob.glob(directory + '**/*.png', recursive=True)
    n_imgs = len(img_paths) # number of images
    imgs = numpy.zeros((n_imgs,h,w,dp), dtype=numpy.float32)
    
    
    listfile =[]
    listot =[]
    labels=[]
    j=0
    for root, dirnames, filenames in os.walk(directory):
        for dirname in dirnames:
            addrs = glob.glob(root + dirname + '/*.png')
            for i, val in enumerate(addrs):
                #labels.append(len(listfile)+1)
                labels.append(dirname)
                img = imread(val)
        
                img = imresize(img,(h,w))    
                
                imgs[j, ...] = img
                #imgs[j, ...] = cv2.logPolar(img, (0, 0), 40, cv2.WARP_FILL_OUTLIERS)
                j=j+1
            listfile.append(addrs)
            print('read files:' + root + dirname)
    
    #labels=LabelBinarizer().fit_transform(labels)        
    for item in listfile:
        listot = listot + item
        
    return imgs,labels


base_dir = 'TUDarmstadt_fred/PNGImages/'
X,Y=readimage(base_dir)

base_dir = 'TUDarmstadt_fred/PNGImages_mix4/'
X1,Y1=readimage(base_dir)
########################### ENCODING CLASSES ################################
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

### Change any labels to sequential integer labels
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
Yn = encoder.transform(Y)

encoder.fit(Y1)
Y1n = encoder.transform(Y1)

# convert integers to dummy variables (i.e. one hot encoded)
Y_onehot = np_utils.to_categorical(Yn)
Y1_onehot = np_utils.to_categorical(Y1n)



########################### NORMALIZATION ####################################
X /= 255 # ou gaussian
X1 /= 255 # ou gaussian

### Split training and test
#X_train, X_test, y_train, y_test = train_test_split(X, Yi, test_size=0.20, random_state=42)
X_train,y_train=X,Y_onehot
X_test,y_test=X1,Y1_onehot

############################################ CONVOLUTIONAL NEURAL NETWORK ######################################################

# Import libraries and modules
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from logpolar import LogPolar

# Load data into train and test sets

# Preprocess input data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Preprocess class labels
Y_train = y_train
Y_test = y_test

def build_cnn():
    # Define model architecture
    model = Sequential()
    # asift layer
    # logpolar layer
    model.add(LogPolar(input_shape=(h, w, dp)))
    model.add(Conv2D(32, (3, 3), activation='relu'))#, input_shape=(h, w, dp)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # 200x150 => 100x75
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = build_cnn()

# Fit model on training data
model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

print(model.metrics_names)
# Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)

y_predv = model.predict(X_test)

