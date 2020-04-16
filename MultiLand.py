'''
The following code references https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
tensorflow's own guide to image classification.
Adapting to EuraSat labelled data set from 
'''
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import matplotlib.pyplot as plt
import IPython.display as display
from PIL import Image
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
import numpy as np
import cv2
import pathlib
import math
import tensorflowjs as tfjs

# https://www.tensorflow.org/guide/gpu
# manage memory growth running on GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# checking computing device, showing if computation is done from GPU and CPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Process image to be loaded

image_count = 27000

BATCH_SIZE = 32
IMG_HEIGHT = 64
IMG_WIDTH = 64
STEPS_PER_EPOCH = np.ceil(image_count / BATCH_SIZE)

DATADIR = "EuroSAT//2750"
CLASS_NAMES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", 
              "Industrial", "Pasture", "PermanentCrop", "Residential", 
              "River", "SeaLake"]    

# Load pickled data *Eurosat dataset
import pickle
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

# Normalize Image data
X = X / 255.0 
# Convert to numpy array
X = np.array(X)
#X = np.array(X).reshape(-1, 64, 64, 3)
y = np.array(y)

print('Inspecting shape of inputs and labels')

IMG_SIZE = 64
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

print('Initializing base model')
# Following image classfier from https://www.tensorflow.org/tutorials/images/classification
input_layer = tf.keras.layers.Conv2D(16, 3,input_shape=(64, 64, 3))
max_pooling2d = tf.keras.layers.MaxPool2D()
conv2d_16 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')
conv2d_32 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')
conv2d_64 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')
dropout_layer = tf.keras.layers.Dropout(0.2)
dense_784 = tf.keras.layers.Dense(784, activation='relu')
flatten_layer = tf.keras.layers.Flatten()
prediction_layer = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

model = tf.keras.Sequential([
  input_layer,
  max_pooling2d,
  dropout_layer,
  conv2d_32,
  max_pooling2d,
  conv2d_64,
  dropout_layer,
  flatten_layer,
  dense_784,
  prediction_layer
])

print('Compiling pretrained model')
base_learning_rate = 1e-3
model.compile(optimizer=tf.keras.optimizers.RMSprop(base_learning_rate),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
  metrics=['accuracy'])

class_freq = [3000, 3000, 3000, 2500, 2500, 2000, 2500, 3000, 2500, 3000]
  
class_freq = [freq / image_count for freq in class_freq]

# all equal weight
class_weight = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0,
                5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}

for key, val in class_weight.items():
  class_weight[key] = class_freq[key]

print(class_weight)

print(y)

history = model.fit(X, y, 
  class_weight=class_weight, 
  batch_size=32, 
  epochs=1000, 
  validation_split=0.2, 
  verbose=1)

IMAGE_SHAPE = (64, 64)
imArray = cv2.imread('sample.jpg')
imArray = cv2.resize(imArray, IMAGE_SHAPE)

test = np.array(imArray).reshape(-1, 64, 64, 3)
test = test / 255.0

# Save historing data in history pickle
import pickle
pickle_out = open("history1.pickle", "wb")
pickle.dump(history.history, pickle_out)
pickle_out.close()

output = model.predict(test)
print(output)
print(CLASS_NAMES[output.argmax()])

model.save('models\TestNet4')

tfjs.converters.convert_tf_saved_model('models\TestNet4', 'jsmodels\TestNet4')