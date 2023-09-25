#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 16:05:31 2023


We want to train the overall symreg to recognize a density, and its transformed counterpart

This function takes densities, applies an operation as we discussed in meetings 
(+, *, sin, cos, tan, exp, diff, integ) and then tags that. It will be fed into 
a group of discriminator CNN models to identify if the operation was made or not. 
"""
import os
import numpy as np
import scipy
import random
import pickle
binaries = ["add","sub","mul","div","pow"]
unaries = ["diff","inte","cos","sin",'tan',"log"]
symvect = [0 for i in unaries+binaries]

import pickle

def power(ar,k):
    return pow(ar,k)
def diff(ar):
    delt = np.diff(ar,n=1)
    delt.resize(len(delt)+1)
    return delt
def inte(ar):
    upto = 0
    AR = []
    for i in range(len(ar)):
        upto+=ar[i]
        AR.append(upto)
    return np.array(AR)
def add(ar,b):
    #add some random const
    return ar+b
def mul(ar,b):
    #mul some random const
    return ar*b
def sin(ar):
    #apply sin(const+ar) to account for cos as well
    return np.sin(ar+np.random.randint(-157,157)/100)
def exp(ar):
    return np.exp(ar)
def log(ar):
    return np.log(ar)
def iden(ar):
    return ar
def tan(ar):
    return np.tan(ar)
operations = ["add","mul","inte","diff","sin","exp","log"]
import random_expressions
cwd = "/home/randon/Documents/homework/2020/mcgill/rustam/next-project/software/data/"
numattractors = 1
for i in range(numattractors):
    wcwd = cwd+str(i+1)+"/"
    for file in os.listdir(wcwd):
        if "-data.txt" in file:
            with open(wcwd+file,"r") as f:
                print(file)
                contents = f.read()
                parts = contents.split("=")
                energies = parts[1]
                energies = energies.replace("\nwfns ","")
                energies = eval(energies)
                wfns = parts[2]
                wfns = wfns.replace("\nKEDs ","")
                wfns = wfns.replace("array","np.array")
                wfns = eval(wfns)
                KEDs = parts[3]
                KEDs = KEDs.replace("\nv ","")
                KEDs = KEDs.replace("array","np.array")
                KEDs = eval(KEDs)
                v = parts[4]
                v = v.replace("\nsymrep ","")
                v = v.replace("\n","")
                v = v.replace(" ",",")
                v = v.replace("[","")
                v = v.replace("]","")
                v = v.split(",")
                v = np.array([float(i) for i in v if len(i)>0])
                symrep = parts[5].replace("\n","")
                #v = np.array(q)
                #v = eval(v)
                """This will create the variables 
                energies: dict(int:[float]) - the energy of each solution
                wfns: dict(int:[np.array([float])]) - wavefunction for each eigenstate
                KEDs: dict(int:[np.array([float])]) - kinetic energy density of each eigenstate
        """
        syntree, expr = random_expressions.create_random_expressions(1,operations,unaries,binaries,maxops=2)
        k = random.randint(-1000,1000)/100 #shows up in expr
        expr = expr[0]
        expr = expr.replace("x","wfns[0][0]")
        transformed = eval(expr)
        symvectk = unaries+binaries
        for idx,k in enumerate(symvectk):
            if k in expr:
                symvect[idx] = 1
        """data out:
            {wf: orig_wf
            symrepV: str for v symrep
            vector for the transforms used: symvect
            transformed: transformed}
            """
        data = {"wf":wfns,"symrepV":symrep,"symvect":symvect,"transformed":transformed,"V":v,"E":energies,"KEDs":KEDs}
        with open(cwd+"transforms/"+str(i+1)+"/"+file+".pckl","wb") as f:
            pickle.dump(data,file=f)
    