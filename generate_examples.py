#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 02:02:53 2023

@author: randon
"""

import numpy as np
from numba import jit
import scipy.integrate
import matplotlib.pyplot as plt
import os
#configuration
ngrids=1000

L = 40
dx = 1
hbar = 1
m = 1
g2 = ((2*m)*(L**2))/(hbar**2) #contains all the constants that stay out of the way
l2 = (1/(ngrids-1))**2 #length**2

def normalize(wfn):
    #Normalization actually doesnt matter until the final step because the energy
    #stability in the numerov method is scale-invariant. 
    Ainv = wfn**2
    Ainv = scipy.integrate.simps(Ainv,dx=1)
    #print(scipy.integrate.simps((wfn/np.sqrt(Ainv))**2))
    return wfn/np.sqrt(Ainv)
@jit
def wf_nm(Psi: np.array([np.float]),k2: np.array([np.float]),N:int) -> np.array([np.float]):
    #k2[x]: g2*(E-V[x])
    for i in range(2,N):
        Psi[i]= (2*(1-(5.0*(dx**2)/12)*l2*k2[i-1])*Psi[i-1] - (1+((1.0*(dx**2))/12)*l2*k2[i-2])*Psi[i-2])/(1+((dx**2)/12)*l2*k2[i]) 
    return Psi

def converge(eps,deps,v,N=ngrids):
    #eps: first guess energy
    #deps: delta eps to climb up on
    #N: num grid points
    Psi = np.zeros(N)
    Psi[0] = 0
    Psi[1] = 1e-4
    k2 = g2*(eps-v)
    wfn = wf_nm(Psi,k2,N)
    P1 = wfn[N-1]
    eps = eps+deps
    while abs(deps)>1e-12:
        #print(deps)
        k2 = g2*(eps-v)
        wfn = wf_nm(wfn,k2,N)
        P2 = wfn[N-1]
        if P1*P2<0:
            deps = -deps/2
        eps = eps+deps
        P1 = P2
    
    return eps,normalize(wfn)

#creating random functions and their potentials here
import random
def generate_potentials(L,maxattractors=3,maxwidth=30):
    """
    generate a bunch of potentials of the form 
    -\sum_N A_i*exp((x-b_i)**2)
    and returns a callable function, and their symbolic form."""
    symbolicrep = "0 "
    numattract = random.randint(1,maxattractors)
    maxwidth = maxwidth*L/100
    for pot in range(numattract):
        #displacement wont go wild and off the range of our window
        displacement = L*maxwidth*maxattractors*0.015*random.randint(-10000,10000)/10000
        
        depth = random.randint(0,100)/100
        #ditto for width wont get too wide and leave the window.
        stddev = (L/maxwidth)*random.randint(10,50)/100
        
        symbolicrep += "- "+str(depth)+"*"+"np.exp(-(x-"+str(displacement)+")**2/(2*"+str(stddev)+"**2))"
        #print(symbolicrep)
    return lambda x: eval(symbolicrep), symbolicrep, numattract


def KED(wfn):
    """"There are theoretical reasons why |Psi'(x)|^2 is preferred over |Psi(x)*Psi''(x)| """
    delt = np.diff(wfn,n=1)
    delt.resize(len(delt)+1) #pad 2 to deal with diff twice
    return delt*delt

maxexamples = 20
for j in range(maxexamples):
  pot,symrep,minimae = generate_potentials(L*0.5)
  levels = []
  wfns = {0:[],1:[],2:[],}
  energies = {0:[],1:[],2:[],}
  KEDs = {0:[],1:[],2:[],}
  v = np.vectorize(pot)(np.linspace(-L/2,L/2,ngrids))
  for i in range(minimae):
      
      if i==0:
          eguess = min(v)
      else:
          eguess= energies[i-1][0]
      E, wfn = converge(eguess+0.001,1e-1,v)
      if E<0:
          
          energies[i].append(E)
          wfns[i].append(wfn)
          #plt.plot(wfn)
          ked = KED(wfn)
          KEDs[i].append(ked)
      else:
          print("unbound!")
          break

  if j%10==0:
      print(j)
  with open(os.getcwd()+"/data/"+str(minimae)+"/"+str(j)+"-data.txt","w") as f:
    outstr = """energies = {{ENERGIES}}
wfns = {{WFNS}}
KEDs = {{KEDS}}
v = {{V}}
symrep = {{SYMREP}}
    """
    outstr = outstr.replace("{{ENERGIES}}",str(energies))
    outstr = outstr.replace("{{WFNS}}",str(wfns))
    outstr = outstr.replace("{{KEDS}}",str(KEDs))
    outstr = outstr.replace("{{V}}",str(v))
    outstr = outstr.replace("{{SYMREP}}",str(symrep))

    f.write(outstr)
    