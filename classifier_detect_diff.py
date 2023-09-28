#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 20:58:53 2023

@author: randon

A series of classifiers to detect features. 
This one seeks to detect differentiation
"""


from tensorflow import keras
import tensorflow as tf
import numpy as np
import pickle
import os
import dataGenerator
import keras.layers


from tensorflow.keras import layers

"""begin model definition here"""
#x = layers.Dense(units=1000,activation="relu",name="dense1")(inputs)
"""
Good architecture is 2 1d CNN followed by dropout layer for regularization then a pooling layer. 

For the CNN: each input is 2000 timesteps (wfn+transform), with 1 signal (y(x))


"""
#x = layers.Conv1D(filters=32, kernel_size=(1), activation="relu")(inputs)
#x = layers.Conv1D(filters=32, kernel_size=(1), activation="relu")(x)
#x = layers.Flatten()(x)

model = keras.Sequential()
model.add(layers.Dense(2000,input_shape=(2000,), name="input"))
model.add(layers.Reshape( (2000,1),input_shape=(2000,) ) )
model.add(layers.Conv1D(filters=32,kernel_size=50,activation="relu"))
model.add(layers.Conv1D(filters=32,kernel_size=50,activation="relu"))
model.add(tf.keras.layers.AveragePooling1D(
    pool_size=5, strides=None, padding="valid", data_format="channels_last",
)
)
model.add(layers.Flatten())
model.add(layers.Dense(2000, name="dense1",activation="relu"))
model.add(layers.Dense(1,    name="dense3",activation='sigmoid'))


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.FalseNegatives()],
)
train_generator = dataGenerator.example()
res = [ex for ex in train_generator]
x = np.array([ex[0] for ex in res])
y = np.array([ex[1] for ex in res])
model.fit(
    x=x,
    y=y,
    epochs=40,
    )