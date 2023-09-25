#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 20:58:53 2023

@author: randon

A series of classifiers to detect features. 
This one seeks to detect differentiation
"""


from tensorflow import keras
import numpy as np
import pickle
import os
import sklearn.preprocessing
cwd = "/home/randon/Documents/homework/2020/mcgill/rustam/next-project/software/data/transforms/1"
binaries = ["add","sub","mul","div","pow"]
unaries = ["diff","inte","cos","sin",'tan',"log"]
symvectk = unaries+binaries
def example():
    """returns x_train, y_train
    """
    for file in os.listdir(cwd):
        if ".pckl" in file:
            with open(cwd+"/"+file,"rb") as f:
                res = pickle.load(f)
                #normalizing the wf, transformed wf to be zero-mean and unit variance
                #this is needed as blowups can often occur.
                wfn = res['wf'][0][0]
                wfn = wfn/(max(wfn)-min(wfn))
                wfn = wfn-np.average(wfn)
                transform = res['transformed']
                transform = transform/(max(transform)-min(transform))
                transform = transform-np.average(transform)
        yield np.array((wfn,transform)), symvectk.index('diff')

from tensorflow.keras import layers

"""begin model definition here"""
inputs = keras.Input(shape=(2,1000))
x = layers.Conv1D(filters=32, kernel_size=(1), activation="relu")(inputs)
x = layers.Conv1D(filters=32, kernel_size=(1), activation="relu")(x)
x = layers.Flatten()(x)
num_classes = 1
outputs = layers.Dense(units=1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
