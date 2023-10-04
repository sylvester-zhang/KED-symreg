#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 17:54:46 2023

@author: randon
given a CNN, try to generate the KED using only a CNN approach.

KEDs are normalized to have range [-1,1] and mean=0. 
"""

from tensorflow import keras
import tensorflow as tf
import numpy as np
import pickle
import os
import dataGenerator
import keras.layers
import scipy

from tensorflow.keras import layers
def normalize(wfn):
    #Normalization actually doesnt matter until the final step because the energy
    #stability in the numerov method is scale-invariant. 
    Ainv = wfn**2
    Ainv = scipy.integrate.simps(Ainv,dx=1)
    #print(scipy.integrate.simps((wfn/np.sqrt(Ainv))**2))
    return wfn/np.sqrt(Ainv)

def KED(wfn):
    """"There are theoretical reasons why |Psi'(x)|^2 is preferred over |Psi(x)*Psi''(x)| """
    delt = np.diff(wfn,n=1)
    delt.resize(len(delt)+1) #pad 2 to deal with diff twice
    return delt*delt


train_generator = dataGenerator.example()
res = [ex for ex in train_generator]
xy = [ex[0] for ex in res]
x = np.array([i[:1000] for i in xy])
y = np.array([normalize(KED(i)) for i in x])


model = keras.Sequential()
model.add(layers.Dense(1000,input_shape=(1000,), name="input"))
model.add(layers.Reshape( (1000,1),input_shape=(1000,) ) )
model.add(layers.Conv1D(filters=32,kernel_size=50,activation="relu"))
model.add(layers.Conv1D(filters=32,kernel_size=50,activation="relu"))
model.add(tf.keras.layers.AveragePooling1D(
    pool_size=10, strides=None, padding="valid", data_format="channels_last", 
    #bigger pool size results in less data being passed through the pooling layer, 
    #for better translational invariance, but potentially results in too much data lost
)
)
model.add(layers.Flatten())
model.add(layers.Dense(1000, name="dense1",activation="relu"))
model.add(layers.Dense(1000, name="output",activation="tanh")) #ys are normalized, leakyReLU allows for range from [-inf,inf]


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.MeanSquaredError(),
)

model.fit(
    x=x,
    y=y,
    epochs=40,
    )
