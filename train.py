#!/usr/bin/env python 
"""
A python script used for Term 1, Project 3 Behavioral Cloning
This script does data ingestion and training
"""

import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D
from keras.layers.convolutional import Cropping2D
import matplotlib.pyplot as plt

# Globals for Training/Testing
EPOCHS = 3
CORRECTION_FACTOR = .2
BATCH_SIZE = 36 # This number must be divisible by 6, because I sample each line 6 times

def get_filename(path):
    """
    a helper function to get the filename of an image. 
    This function splits a path provided by the argument by '/'
    and returns the last element of the result
    """
    return path.split('/')[-1]

def generator(samples, batch_size = BATCH_SIZE):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            lines_to_process = batch_size/6
            batch_samples = samples[offset:offset+ int(lines_to_process)]

            augmented_images, augmented_measurements = [],[]
            for batch_sample in batch_samples:
                # Load the images
                path = './data/IMG/' # The current path of where the data is located
                center_image = cv2.imread(path + get_filename(line[0]))
                left_image = cv2.imread(path + get_filename(line[1]))
                right_image = cv2.imread(path + get_filename(line[2]))
                # Load the measurements associated with these images
                measurement = float(line[3])
                left_measurement = measurement + CORRECTION_FACTOR
                right_measurement = measurement - CORRECTION_FACTOR
                # capture the images for the center, left and right cameras
                augmented_images.extend([center_image, left_image, right_image])
                augmented_measurements.extend([measurement, left_measurement, right_measurement])
                # and the flipped image, so we get twice the data for free
                augmented_images.extend([cv2.flip(center_image, 1), cv2.flip(left_image, 1), cv2.flip(right_image, 1)])
                augmented_measurements.extend([measurement * -1.0, left_measurement * -1.0, right_measurement * -1.0] )

            # Put the data into numpy arrays so that keras can use it
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield shuffle(X_train, y_train)


lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size = 0.2)

train_generator = generator(train_samples, batch_size = BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size = BATCH_SIZE)

#Instantiate the model
model = Sequential()
#Normalize the data
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3))) #normalize the data and give it a mean of 0
# Crop the data
model.add(Cropping2D(cropping=((70,25),(0,0))))
# Nvidia model taken from: https://devblogs.nvidia.com/deep-learning-self-driving-cars/
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
# compile the model
model.compile(loss='mse', optimizer='adam')

# train the model
model.fit(X_train, y_train, validation_split=.2, shuffle=True, nb_epoch=EPOCHS)

#history_object = model.fit_generator(train_generator, samples_per_epoch =
#    len(train_samples), validation_data = 
#    validation_generator,
#    nb_val_samples = len(validation_samples), 
#    nb_epoch=EPOCHS, verbose=1)

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()