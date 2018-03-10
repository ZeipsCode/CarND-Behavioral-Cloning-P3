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

with open('/media/zpr/5aa7062e-a1a2-4b29-85cb-5756318d57ee/Udacity/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
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
        current_path = '/media/zpr/5aa7062e-a1a2-4b29-85cb-5756318d57ee/Udacity/CarND-Behavioral-Cloning-P3/data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
    n+=1



data_folder = '/media/zpr/5aa7062e-a1a2-4b29-85cb-5756318d57ee/Udacity/CarND-Behavioral-Cloning-P3/data/'

log_path = data_folder + 'driving_log.csv'
logs = []

delta = 0.08



with open(log_path,'rt') as f:
    reader = csv.reader(f)
    for line in reader:
        logs.append(line)
    log_labels = logs.pop(0)

    for i in range(len(logs)):
        img_path = logs[i][0]
        img_path = data_folder+'IMG'+(img_path.split('IMG')[1]).strip()
        measurements.append(float(logs[i][3]))
    
    for i in range(len(logs)):
        img_path = logs[i][1]
        img_path = data_folder+'IMG'+(img_path.split('IMG')[1]).strip()
        measurements.append(float(logs[i][3]) + delta)
    
    for i in range(len(logs)):
        img_path = logs[i][2]
        img_path = data_folder+'IMG'+(img_path.split('IMG')[1]).strip()
        measurements.append(float(logs[i][3]) - delta)



augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement * -1.0)

    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)    


#print(np.sum(measurements))    
#print(np.sum(augmented_measurements))




from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D, Dropout

model = Sequential()
model.add(Lambda(lambda x: x/255 -0.5,input_shape=(160,320,3) ))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(12, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
#model.add(Convolution2D(3, (2, 2), activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
#model.add(Dense(10))
model.add(Dense(1), activation = 'tanh')
model.summary()

model.compile(loss = 'mse', optimizer = 'adam' , metrics=['accuracy'])
model.fit(X_train, y_train, validation_split = 0.1, shuffle = True, nb_epoch = 20, batch_size = 128)

model.save('model_augmented_images_20_Epochs.h5')

### nochmal länger trainieren! lohnt sich
