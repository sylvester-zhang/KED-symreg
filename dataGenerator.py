#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 00:37:24 2023

@author: randon
"""
import keras
import os
import pickle
import numpy as np
binaries = ["add","sub","mul","div","pow"]
unaries = ["diff","inte","cos","sin",'tan',"log"]
symvectk = ['diff', 'inte', 'cos', 'sin', 'tan', 'log', 'add', 'sub', 'mul', 'div', 'pow']
cwd = os.getcwd()+"/data/transforms/1"
filesavail = os.listdir(cwd)


np.seterr(all='raise')

def example(cwd=cwd,filesavail=filesavail):
    """returns x_train, y_train
    """
    print(cwd)
    print(os.listdir(cwd))
    
    for file in filesavail:
        if ".pckl" in file:
            print(file)
            with open(cwd+"/"+file,"rb") as f:
                res = pickle.load(f)
                #normalizing the wf, transformed wf to be zero-mean and unit variance
                #this is needed as blowups can often occur.
                print(res['symvect'][symvectk.index('diff')],res['transformation'])

                try:
                    wfn = res['wf'][0][0]
                except IndexError as ie:
                    if len(res['wf'][0])==0:
                        os.remove(cwd+"/"+file)
                        pass
                wfn = wfn/(max(wfn)-min(wfn))
                wfn = wfn-np.average(wfn)
                transform = res['transformed']
                transform = np.nan_to_num(transform,nan=0.0)
                if type(max(transform))==type(min(transform)):
                    if type(min(transform))==np.float64:
                        try:
                            print(max(transform),min(transform))
                            transform = transform/((max(transform)-min(transform))+1e-5)
                            
                            transform = transform-np.average(transform)
                        except FloatingPointError as fpe:
                            print(fpe,"deleting",file)
                            os.remove(cwd+"/"+file)
                            pass
                        
        yield np.concatenate((wfn,transform)), np.array( [ res['symvect'][symvectk.index('diff')] ] )

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, cwd=cwd, batch_size=1, dim=(2000), n_channels=1,
                 n_classes=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.cwd = cwd
    def __len__(self):
        'Denotes the number of batches per epoch'
        return 1

    def __getitem__(self, cwd):
        'Generate one batch of data'
        # Generate indexes of the batch
        
        # Generate data
        X, y = self.__data_generation()

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(filesavail)

    def __data_generation(self):
        return next(example(self.cwd,filesavail))