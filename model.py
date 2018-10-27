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
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Dropout, SpatialDropout2D, Cropping2D
import numpy as np

#Globals
CORRECTION_FACTOR = 0.2 # How much to correct our steering measurement
DEBUG = False # Whether or not to print out debug information.  This will slow down the program significantly
EPOCHS = 5 # Number of Epochs to train the model
STRAIGHT_THRESHOLD = 0.0 # The steering measurement threshold where we drop
STRAIGHT_DROP_PROB = .9 # Amount of straight samples to drop
MODEL = 2 # The keras model to use. 1 is Lenet, 2 is NVIDIA

def get_filename(path):
    """
    a helper function to get the filename of an image. 
    This function splits a path provided by the argument by '/'
    and returns the last element of the result
    """
    return path.split('/')[-1]

def left_steering(measurement):
    """
    a helper function to make sure we do not over correct
    """
    measurement = (measurement + CORRECTION_FACTOR)
    return measurement

def right_steering(measurement):
    """
    a helper function to make sure we do not over correct
    """
    measurement = (measurement - CORRECTION_FACTOR)
    return measurement

def remove_straights(samples, drop_prob = STRAIGHT_DROP_PROB, threshold = STRAIGHT_THRESHOLD):
    """
    a helper function to remove a percentage(keep_prob) of the samples that have a steering near 0
    This is to prevent overfitting of straights
    """
    i = 1 # Start at one, since we are manipulating array indices
    num_of_deleted_samples = 0
    while i < len(samples):
        measurement = samples[i][3]
        if abs(float(measurement)) <= threshold:
            if np.random.rand() < drop_prob:
                if DEBUG:
                    print("Deleting Sample: " + str(samples[i][0]) + " Measurement of: " + str(samples[i][3]))
                del samples[i]
                i -= 1
                num_of_deleted_samples += 1
        i += 1
    print("Deleted " + str(num_of_deleted_samples) + " Samples")
    return samples

#Main
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)
    for line in reader:
        lines.append(line)

samples = remove_straights(lines)

images = []
measurements = []
for line in samples:
    for i in range(0,3):
        local_path = "./data/IMG/" + get_filename(line[i])
        image = mpimg.imread(local_path)
        images.append(image)
        if DEBUG: 
            print(local_path)
    # Append Center measurement
    measurement = float(line[3])
    measurements.append(measurement)
    # Append Left Measurement
    measurements.append(left_steering(measurement))
    # Append Right Measurement
    measurements.append(right_steering(measurement))
    if DEBUG:
        print("Center Measurement" + str(measurements[0]))
        print("Left Measurement" + str(measurements[1]))
        print("Right Measurement" + str(measurements[2]))

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
# Lenet Model
if MODEL is 1:
    print("Using Model: LENET")
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - .5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(6,5,5, activation= "relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(16,5,5, activation= "relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
# NVIDIA Model
if MODEL is 2:
    print("Using Model: NVIDIA")
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - .5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24,5,5,subsample=(2,2), activation= "relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2), activation= "relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2), activation= "relu"))
    model.add(Convolution2D(64,3,3, activation= "relu"))
    model.add(Convolution2D(64,3,3, activation= "relu"))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))


# compile the model
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, epochs=EPOCHS)
model.save('model.h5')
keras.backend.clear_session()