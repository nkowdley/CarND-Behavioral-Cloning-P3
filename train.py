#!/usr/bin/env python 
"""
A python script used for Term 1, Project 3 Behavioral Cloning
This script does data ingestion and training
"""

import csv
import cv2
import matplotlib.image as mpimg
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Dropout, SpatialDropout2D
from keras.layers.convolutional import Cropping2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from pprint import pprint

# Globals and Hyperparameters for Training/Testing
EPOCHS = 5
CORRECTION_FACTOR = .05
BATCH_SIZE = 128 
STRAIGHT_KEEP_PROB = .8
STRAIGHT_THRESHOLD = .1
LEARNING_RATE = 0.001

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
    if measurement > 1:
        measurement = 1
    if measurement < -1:
        measurement = -1
    return measurement

def right_steering(measurement):
    """
    a helper function to make sure we do not over correct
    """
    measurement = (measurement - CORRECTION_FACTOR)
    if measurement < -1:
        measurement = -1
    if measurement > 1:
        measurement = 1
    return measurement

def remove_straights(samples, drop_prob = STRAIGHT_KEEP_PROB, threshold = STRAIGHT_THRESHOLD):
    """
    a helper function to remove a percentage(keep_prob) of the samples that have a steering near 0
    This is to prevent overfitting of straights
    """
    i = 1
    while i < len(samples):
        measurement = samples[i][3]
        if abs(float(measurement)) < threshold:
            if np.random.rand() < drop_prob:
                del samples[i]
                i -= 1
        i += 1
    return samples

def generator(samples, batch_size = BATCH_SIZE):
    """
    a generator to help efficiently allocate memory
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            augmented_images, augmented_measurements = [],[]
            for batch_sample in batch_samples:
                # Load the images
                path = './data/IMG/' # The current path of where the data is located
                #path = './edata/IMG/'
                center_image = mpimg.imread(path + get_filename(line[0]))
                left_image = mpimg.imread(path + get_filename(line[1]))
                right_image = mpimg.imread(path + get_filename(line[2]))
                # Load the measurements associated with these images
                measurement = float(line[3])
                left_measurement = left_steering(measurement)
                right_measurement = right_steering(measurement)
                # capture the images for the center, left and right cameras
                augmented_images.extend([center_image, left_image, right_image])
                augmented_measurements.extend([measurement, left_measurement, right_measurement])
                # and the flipped image, so we get twice the data for free
                augmented_images.extend([np.fliplr(center_image), np.fliplr(left_image), np.fliplr(right_image)])
                # note that flipped images turn the opposite direction, so recalculate measurements
                measurement = measurement * -1.0
                left_measurement = right_steering(measurement)
                right_measurement = left_steering(measurement)
                augmented_measurements.extend([measurement, left_measurement, right_measurement] )

            # Put the data into numpy arrays so that keras can use it
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield shuffle( X_train, y_train )


lines = []
with open('./data/driving_log.csv') as csvfile:
#with open('./edata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Strip out some of the data
remove_straights(lines)

train_samples, validation_samples = train_test_split(lines, test_size = .20)

train_generator = generator(train_samples, batch_size = BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size = BATCH_SIZE)

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
model.compile(loss='mae', optimizer=Adam(lr = LEARNING_RATE))

# train the model
#model.fit(X_train, y_train, validation_split=.2, shuffle=True, nb_epoch=EPOCHS)
print(len(train_samples))
print(len(lines))
model.fit_generator(train_generator, 
    samples_per_epoch= len(train_samples)*6, 
    validation_data=validation_generator, 
    nb_val_samples=len(validation_samples)*6, 
    nb_epoch=EPOCHS)

model.save('model.h5')
# print the keys contained in the history object
#print(history_object.history.keys())

# plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
