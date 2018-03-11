#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 17:17:13 2018

@author: zpr
"""

import csv
import cv2
import numpy as np


lines = []


with open('/Udacity/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
n = 0
for line in lines:
    if n > 0:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = '/Udacity/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
    n+=1

# flip every image before giving it to the network to train

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement * -1.0)

    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)    
    

# Convolutional Network archtecture starts here

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D, Dropout
from keras import optimizers

model = Sequential()
model.add(Lambda(lambda x: x/255 -0.5,input_shape=(160,320,3) )) # normalize image values to range between 0 and 1
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(6, (3, 3), activation='relu')) # relu nonlinearity as activation of convolutional layer with kernel size of 3x3
model.add(MaxPooling2D(pool_size=(4, 4))) # maxPooling with 4x4 kernel size to reduze information to the most important aspects
model.add(Dropout(0.25))    # dropout with 25 % probability to prevent overfitting
model.add(Flatten())
model.add(Dense(1, activation = 'tanh'))    # fullyConnected layer as output with tanh activation to get values between 0 and 1
model.summary()

model.compile(loss = 'mse', optimizer = 'Adam' , metrics=['accuracy'])
model.fit(X_train, y_train, validation_split = 0.2, shuffle = False, nb_epoch = 2, batch_size = 32)

model.save('model.h5')
