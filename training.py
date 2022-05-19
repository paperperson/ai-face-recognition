import tensorflow as tf
import keras
from tensorflow.keras import layers
from keras.layers import Conv2D
from tensorflow.keras.models import Sequential
import random
from tensorflow.keras.optimizers import Adam

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K

img_height, img_width = 200, 200
batch_size = 32
img_channels = 1

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "D:/train_pics/",
    validation_split=0.2,
    label_mode='categorical',
    subset="training",
    color_mode='grayscale',
    seed=1233,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "D:/train_pics/",
    validation_split=0.2,
    label_mode='categorical',
    subset="validation",
    color_mode='grayscale',
    seed=1233,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = val_ds.class_names

print(class_names)

num_classes = len(class_names)

# def MmNet(input_shape, output_shape):
#     model = Sequential()
#     model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(32, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Conv2D(32, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(output_shape, activation='softmax'))
#     model.summary()
#     return model

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(img_height, img_width, img_channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer =Adam(lr=1e-4), metrics=['accuracy'])

# def loadCNN(bTraining=False):
#     global get_output
#     model = Sequential()
#
#     model.add(Conv2D(128, (5, 5),
#                      padding='valid',
#                      input_shape=(img_height, img_width, img_channels)))
#     convout1 = Activation('relu')
#     model.add(convout1)
#     model.add(Conv2D(32, (3, 3)))
#     convout2 = Activation('relu')
#     model.add(convout2)
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.5))
#
#     model.add(Flatten())
#     model.add(Dense(128))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(num_classes))
#     model.add(Activation('softmax'))
#
#     # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#     model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
#
#     # Model summary
#     model.summary()
#     # Model conig details
#     model.get_config()
#
#     return model


# model = Sequential([
#   layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_classes)
# ])


epochs = 100
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# model.save('D:/cs final project/me.face.model.h5')
