#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 23:58:02 2023

@author: randon
"""

import numpy as np
from numba import jit
import matplotlib.pyplot as plt

xmin = -4
xmax = 4
ngrids = 10000



@jit
def numerovL(Eguess:float,Vx:np.array([float]),deltax:float):
    """Eguess: float - guess for eigenenergy
    Vx: np.array([floats]) - 1d array of floats representing the potential at each grid point
    deltax: float - the width between each grid point, constant. 
    """
    hbar = 1
    m = 1
    Psi = np.zeros(len(Vx))
    Psi[0] = 0
    Psi[1] = 1e-10
    k2 = ((2*m/(hbar**2))*(Eguess-Vx))
    for i in range(1,len(Vx)-1): #start at 1 because Vx is 0-indexed
        Psi[i+1] = (Psi[i]*(2+(10/12)*(deltax**2)*k2[i]) - Psi[i-1]*(1-((deltax**2)/12)*k2[i-1]))/(1-((deltax**2)*(k2[i+1])/12) )
    return Psi
def numerovR(Eguess:float,Vx:np.array([float]),deltax:float):
    """Eguess: float - guess for eigenenergy
    Vx: np.array([floats]) - 1d array of floats representing the potential at each grid point
    deltax: float - the width between each grid point, constant. 
    """
    hbar = 1
    m = 1
    Psi = np.zeros(len(Vx))
    Psi[ngrids-1] = 0
    Psi[ngrids-2] = 1e-5
    k2 = (2*m/hbar)*(Eguess-Vx)
    for i in range(len(Vx)-2,1,-1): #start at 1 because Vx is 0-indexed
        Psi[i-1] = (Psi[i]*(2+(10/12)*(deltax**2)*k2[i]) - Psi[i+1]*(1-((deltax**2)/12)*k2[i+1]))/(1-((deltax**2)*(k2[i-1])/12) )
    return Psi




def converge(ngrids,Vx,niter=10000,etol=1e-7):
    """This function starts with a guess energy and computes Ψ(n−1) for E, and E+deltaE. 
    If there is a sign change in sign between Ψn−1 in the two cases, then deltaE was too big.
    Then update E(i+1) = E(i)-deltaE(i-1)/2
    This process is repeated until deltaE is close to 0.
    Choosing DeltaE as to not overshoot a higher eigenstate is important, so we keep track of the number of modes in the 
    returned Psi. There should be the same number of modes in the 
    """
    Eguess = min(Vx)*(0.999) #we expect the first eigenstate to be somewhere in the potential well...
    deltax = (xmax-xmin)/ngrids
    Psi0 = numerov(Eguess,Vx,deltax) #first psi0
    num0_firstguess = (np.diff(np.sign(Psi0)) != 0).sum() - (Psi0 == 0).sum()
    rangeY = (max(Vx)-min(Vx))
     #probably arent more than 10 bound states?
    #energies = {0:[Eguess],} TODO: implement multiple state search.
    #deltaEs = {0:[deltaE]} 
    energies = [Eguess]
    deltaEs = [1e-5]
    deltaE = min([rangeY/3000,1e-3]) #hopefully this is finegrained enough to never miss a level.
    for i in range(niter):
        #print(deltaEs)
        Eguess = energies[i]
        num0s = (np.diff(np.sign(Psi0)) != 0).sum() - (Psi0 == 0).sum()
        level = num0s - num0_firstguess
        Psi1 = numerov(Eguess+deltaEs[i],Vx,deltax)
        #check that the number of zeroes remains the same as a heuristic check against 
        #overshooting to another energy level...
        #we set max(deltaE) to be small just in case anyways
        num1s = (np.diff(np.sign(Psi1)) != 0).sum() - (Psi1 == 0).sum()
        #if level in energies.keys():
            #energies[level].append()
        energies.append(Eguess+deltaE)
        P0 = Psi0[ngrids-1]
        P1 = Psi1[ngrids-1]
        Psi0 = Psi1
        if P0*P1 < 0 : #sign change! (or overshot!)
            print("Overshot! DeltaE too big!")
            deltaE = -deltaE/2
            deltaEs.append(deltaE)
        else:
            deltaEs.append(deltaE)
        #else:
        #    #we seem to have hit a different energy level...
        #    energies.update({level:[Eguess+deltaE]})
        #    deltaEs.update({level:[deltaE]})
        #check if we are converging
        if np.abs(deltaE) < etol:
            return Psi1,energies
Vx = np.vectorize(lambda x: x**2)
Vx = Vx(np.linspace(xmin,xmax,ngrids))

