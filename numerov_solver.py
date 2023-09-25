#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 20:41:43 2023

@author: randon
"""


from numba import jit
import matplotlib.pyplot as plt

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 15:28:45 2023

"""
from collections.abc import Callable

getE = []
import Fct_Numerov
import numpy as np


def numerov_wf(potential: Callable, symbolic_form: str):

    vx = np.vectorize(potential)
    xmin = -10
    xmax = 10
    ngrids = int(1e5)

    NumEs = 1
    convtol = 1e-1

    EnergyLevelFound = {} #Defines energy levels that avec been found. Has the structure {0:E0, 1:E1, 2:E2, ...}
    WaveFunctionFound = {} #Defines the wave functions that have been found. Has the structure {0:WaveFunction0, 1:WaveFunction1, ...}
    E_guess_try = {} #Defines the lowest and higest energy levels that have been used so far for each number of nodes. Has the structure {NbrNodes1:[Energy guessed min, Energy guessed max], ...}
    iteration = 1 #Defines the number of iterations
    E_found = list() #A list of the energy levels that have been found (ex: [0,2,3] if the energy levels 0,2 and 3 have been found)
    # Note: All the wave function are a list of tuple and have the following structure: [(x0, \psi(x0), (x1, \psi(x1)), ...]

    #Continue while we don't have the n first energy level
    while not E_found == list(range(NumEs)):
        deltavx = (xmax - xmin) / ngrids
        xar = np.arange(xmin,xmax,deltavx)

        Vxar = vx(xar)
        
        E_guess = Fct_Numerov.E_Guess(EnergyLevelFound,E_guess_try,iteration, min(Vxar)+1e-2)
        print('E_guess: ', E_guess)


        MeetingPoints,end_program,E_guess = Fct_Numerov.MeetingPointsPotential(E_guess, Vxar, xar, E_guess_try)
        #MeetingPoints is where the energy of the guess is closest to the potential; it is required that Eguess
        #is captured within the potential at least somewhere, otherwise the wavefunction will be unbound.
        if end_program:
            break
        #Sets the xmin and xmax for the numerov algorithm to explore
        #This is determined by the points after which psi(x) is expected to go to 0.
        Position_min,Position_max = Fct_Numerov.DetermineMinAndMax(MeetingPoints)

        #WaveFunctionNumerov finds a wavefunction for a given E_guess
        WaveFunction = Fct_Numerov.WaveFunctionNumerov(symbolic_form, E_guess, ngrids, 1e-9, Position_min, Position_max)

        NumberOfNodes,PositionNodes,x_max = Fct_Numerov.NumberNodes(WaveFunction)

        # See if the wavefunction for this energy respects the conditions for
        # a well-behaved wavefunction (integrates to 1, smooth)

        conved = Fct_Numerov.VerifyTolerance(WaveFunction,convtol,E_guess,E_guess_try,NumberOfNodes)

        if conved:
        #   print('Wavefunction found!')
            NumberOfNodesCorrected = Fct_Numerov.CorrectNodeNumber(NumberOfNodes,PositionNodes,x_max,E_guess,E_guess_try)
            EnergyLevelFound.update({NumberOfNodesCorrected:E_guess})
            WaveFunctionFound.update({NumberOfNodesCorrected:WaveFunction})


        E_guess_try = Fct_Numerov.SaveEnergy(NumberOfNodes, E_guess, E_guess_try)
        #print('E_guess_try: ',E_guess_try)

        #Increments the iteration
        print('iterations:',iteration,'\n')
        iteration += 1

        E_found = list()
        for i in EnergyLevelFound.keys():
            E_found.append(i)
        E_found.sort()
        print('Energy level found',EnergyLevelFound)
    return WaveFunction, E_found


import random
def generate_potentials(maxattractors=3):
    #generate a bunch of potentials of the form -\sum_Nminima A_i*exp((x-b_i)**2)
    #and returns a callable function, and their symbolic form.
    symbolicrep = "0 "
    numattract = random.randint(1,maxattractors+1)
    for pot in range(numattract):
        displacement = random.randint(-10000,10000)/10000
        depth = random.randint(0,100)/100
        stddev = random.randint(0,100)/100
        
        symbolicrep += "- "+str(depth)+"*"+"np.exp(-(x-"+str(displacement)+")**2/(2*"+str(stddev)+"**2))"
        #print(symbolicrep)
    return lambda x: eval(symbolicrep), symbolicrep

#TODO: bug in solver that breaks when potential ever goes below 0..
#Single-particle WF
#Multiple fermions: slater determinant of each single particle WF. This would be difficult to plot, but one can plot
#a slice of it in x1,x2 space, along with its KED; see equation 2., 3. from Burke et al. 2003.
def density(wf):
    return np.array(wf)**2
def KED(wf):
    """"Equation 2. from "Testing the kinetic energy functional: Kinetic energy density
        as a density functional, Eunji Sim, Joe Larkin, and Kieron Burke, 2003"""
    return np.array(wf)*np.diff(np.diff(np.array(wf)))

maxexamples = 1
for i in range(maxexamples):
  pot,symrep = generate_potentials()
  wf, E = numerov_wf(pot,symrep)
  rwf = [x[1] for x in wf]
  plt.plot(rwf)
  ked = KED(rwf)
  with open(symrep+"-wf.txt","w") as f:
    f.write(wf)
  with open(symrep+"-pot.txt","w") as f:
    f.write(pot)
  with open(symrep+"-KED.txt","w") as f:
    f.write(ked)
