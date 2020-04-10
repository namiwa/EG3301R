from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as pyplot
import numpy as np
import os
import cv2
import pickle
import tensorflowjs as tfjs

# Load pickled data *Eurosat dataset
import pickle
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))
X = np.array(X).reshape(-1, 64, 64, 3)
y = np.array(y)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

print(X[:2])
print(y[:2])

IMG_SIZE = 64
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Set data batches for training
image_count = 27000
BATCH_SIZE = 32
IMG_HEIGHT = IMG_SIZE
IMG_WIDTH = IMG_SIZE
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
CLASS_NAMES = np.array(["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", 
              "Industrial", "Pasture", "PermanentCrop", "Residential", 
              "River", "SeaLake"])   



base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(IMG_SHAPE), include_top=False, weights='imagenet');
base_model.trainable = False;

base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer,
])

model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.add(tf.keras.layers.Activation('relu'))

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

# quick way to train something 
history = model.fit(X, y, batch_size=32, epochs=1, validation_split=0.2)



model.summary()
print(len(model.trainable_variables))

model.save('models\mobileTest2')

tfjs.converters.convert_tf_saved_model('models\mobileTest2', 'jsmodels\mobileTest2')



