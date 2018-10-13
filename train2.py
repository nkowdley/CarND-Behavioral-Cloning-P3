#!/usr/bin/env python 
"""
A python script used for Term 1, Project 3 Behavioral Cloning
This script does data ingestion and training
"""
import csv
import cv2
import matplotlib.image as mpimg
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, MaxPooling2D, Dropout, SpatialDropout2D
import numpy as np

def get_filename(path):
    """
    a helper function to get the filename of an image. 
    This function splits a path provided by the argument by '/'
    and returns the last element of the result
    """
    return path.split('/')[-1]

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    local_path = "./data/IMG/" + get_filename(line[0])
    image = mpimg.imread(local_path)
    images.append(image)
    measurement = line[3]
    measurements.append(measurement)

augmented_images = []
augmented_measurements = []

for image, measurement in zip(images, measurements):
    #standard images
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = float(measurement) * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

X_train = np.array(augmented_images)
Y_train = np.array(augmented_measurements)

#Instantiate the model
model = Sequential()
#Normalize the data
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3))) #normalize the data and give it a mean of 0
# Crop the data
model.add(Cropping2D(cropping=((50,25),(0,0))))
# Nvidia model taken from: https://devblogs.nvidia.com/deep-learning-self-driving-cars/
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
#model.add(SpatialDropout2D(.2))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
#model.add(SpatialDropout2D(.2))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
#model.add(SpatialDropout2D(.2))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
#model.add(Dropout(.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))
# compile the model
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.save('model.h5')