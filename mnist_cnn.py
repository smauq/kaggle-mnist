#!/usr/bin/python

import numpy as np
import sys
import cv2

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

# Input image dimensions
img_rows, img_cols = 28, 28
nb_classes = 10

# Load the training data
X_train = np.genfromtxt('train.csv', delimiter=',')
X_train = X_train[1:]

# Separate features and labels
y_train = X_train[:,0]
Y_train = np_utils.to_categorical(y_train, nb_classes)

X_train = X_train[:,1:].reshape(X_train.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype("float32") / 255

# Load the test data
X_test = np.genfromtxt('test.csv', delimiter=',')
X_test = X_test[1:]
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_test = X_test.astype("float32") / 255

# Initialize network
batch_size = 128
nb_epoch = 16

# Number of convolutional filters to use
nb_filters = 32
# Size of pooling area for max pooling
nb_pool = 2
# Convolution kernel size
nb_conv = 3

# Model definition
model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='full',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1)

# Predict on the test data
y_test = model.predict(X_test, verbose=1)
np.save('predict', y_test)

y_test = np.argmax(y_test,1)
with open('predict.csv', 'w') as fp:
    fp.write('ImageId,Label\n')
    for i,p in enumerate(y_test):
        fp.write('%d,%d\n' % (i+1, p))
