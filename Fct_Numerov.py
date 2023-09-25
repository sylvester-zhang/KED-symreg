
  #!/usr/bin/env python
  #modified from felix deroche's solver; a few bugs fixed with regard to negative potentials, and unneeded code related to
  #interactive use purged.

import numpy as np
import matplotlib.pyplot as plt


def GetFirstEnergyGuess(PotentialArray):


    First_E_guess = PotentialArray.min() +  (1/500000) * (PotentialArray.mean() + PotentialArray.min())

    return First_E_guess


def VerifyConcavity(PotentialArray, First_E_guess):

    i = 1
    #Continue while it doesn't find meeting points
    while i == 1:
        print('First Energy guess:', First_E_guess)
        index_min=list()
        index_max=list()

        #Tries to find meeting points and to compare them
        try:
            for i in range(0,len(PotentialArray)-2):

                #Gets all the points where the potential meets the E_verify value and filters them depending on their derivatives
                if PotentialArray[i] > First_E_guess and PotentialArray[i+1] < First_E_guess:
                    index_min.append(i)

                elif PotentialArray[i] < First_E_guess and PotentialArray[i+1] > First_E_guess:
                    index_max.append(i)

                elif PotentialArray[i] == First_E_guess:
                    if PotentialArray[i-1] > First_E_guess and PotentialArray[i+1] < First_E_guess:
                        index_min.append(i)

                    elif PotentialArray[i-1] < First_E_guess and PotentialArray[i+1] > First_E_guess:
                        index_max.append(i)

            #Defines the concavity value depending on
            #print('index max: ',index_max)
            #print('index_min: ',index_min)

            if (max(index_max) > max(index_min)) and (min(index_max) > min(index_min)):
                concavity = 'positive'
            else:
                concavity = 'negative'

        #If we are not able to compare the potential, we define a new energy guess
        except ValueError:
            First_E_guess = First_E_guess/2

        #If it is able to compare them, exit the loop
        else:
            i = 0

    return concavity,First_E_guess


def EvaluateOnePotential(position,potential):

    x = position
    EvalPotential = eval(potential)

    return EvalPotential

def TranslationPotential(PositionPotential, PotentialArray):

    # i) Gets the minimum value for the potential and the translation in y
    trans_y = PotentialArray.min()
    #index = float(np.where(PotentialArray==trans_y)[0])

    # ii) Defines the necessary translation in x
    #trans_x = x_min + (Div * index)
    #trans_x = PositionPotential[index]

    # iii) Translates the potential
    PotentialArray = PotentialArray - trans_y
    #PositionPotential = PositionPotential - trans_x

    #print('trans_x; ',trans_x)
    print('trans_y; ',trans_y)

    return PositionPotential, PotentialArray

def TranslatePotential(potential,trans_x,trans_y):
    '''Modify the potential expression to center its minimum at x=0 and y=0'''
    #x translation
    #potential = potential.replace('x','(x+' + str(trans_x) + ')')

    #y translation
    potential = potential + '-' +  str(trans_y)

    print(potential)

    return potential


##################################################
# 2) Numerov algorithm functions
# Defines the functions used in the Numerov method
##################################################

#########################
# i) Initial Energy guess

def E_Guess(EnergyLevelFound, E_guess_try, iteration, First_E_guess):

    #print('Iteration: ',iteration)

    #If it is the first time, return the first energy level of the quantum harmonic oscillator
    if iteration == 1:
        E_guess = First_E_guess  #Takes as intial guess the First_E_guess that has previously been defined
        return E_guess

    # I) Define the energy level that we want to find E_level_guess (the lowest energy level that hasn't been found yet)
    #List for the energy that have been found
    Lvl_found = list(EnergyLevelFound.keys())
    Lvl_found.sort()
    #Gets the energy level that we want to find
    E_level_missing = [index for index,Energy in enumerate(Lvl_found) if not Energy <= index]
    if not E_level_missing:
        if not Lvl_found:
            E_level_guess = 0
        else:
            E_level_guess = max(Lvl_found) +1
    else:
        E_level_guess = min(E_level_missing)

    # II) Defining the energy guess depending on the guess that have already been done (E_guess_try)
    #Finds the closest energy energy level (number of nodes) that has been guessed and that corresponds to a smaller or an equal number of nodes than E_level_guess
    try:
        E_level_smaller = max([ E for E in E_guess_try.keys() if E <= E_level_guess ])
    except ValueError:
        E_level_smaller = None
    #Finds the closest energy energy level (number of nodes) that has been guessed and that corresponds to a bigger number of nodes than E_level_guess
    try:
        E_level_bigger = min([ E for E in E_guess_try.keys() if E > E_level_guess ])
    except ValueError:
        E_level_bigger = None

    #Define the energy guess
    #If the smaller and higher exist take the average
    if (not E_level_smaller == None) and (not E_level_bigger ==None):
        E_guess = ( E_guess_try[E_level_smaller][1] + E_guess_try[E_level_bigger][0] ) / 2

    #If only the higher exists take the half
    elif not E_level_bigger == None:
        E_guess = E_guess_try[E_level_bigger][0]/2

    #If only the smaller exists take the double
    elif not E_level_smaller == None:
        E_guess = E_guess_try[E_level_smaller][1] * 2

    print('E_level_guess:', E_level_guess )
    #print('E_level_bigger: ', E_level_bigger)
    #print('E_level_smaller: ', E_level_smaller)

    return E_guess



def MeetingPointsPotential(E_guess, PotentialArray, PositionPotential, E_guess_try):
    p = 1
    iteration = 0
    end_program = False
    while p == 1:
        #Finds all the meeting points
        MeetingPoints = [None,None]
        for i in range(0,len(PotentialArray)-2):
            """scans the potential array for where the potential is at E_guess +- deltaY of potential
            
            """
            deltaY = np.abs(PotentialArray[i] - PotentialArray[i+1])
            if np.abs(PotentialArray[i] - E_guess) < deltaY:
            #(PotentialArray[i] < E_guess and PotentialArray[i+1] > E_guess) or (PotentialArray[i] > E_guess and PotentialArray[i+1] < E_guess) or (PotentialArray[i] == E_guess):
                #And filter them
                if (MeetingPoints[0] == None) or (PositionPotential[i] < MeetingPoints[0]):
                    print('index r min: ',i)
                    MeetingPoints[0] = PositionPotential[i]
                elif (MeetingPoints[1] == None) or (PositionPotential[i] > MeetingPoints[1]):
                    MeetingPoints[1] = PositionPotential[i]
                    print('index r max: ', i)

        #If we have not found at least two meeting points, then make a new smaller (absolute val) energy guess and repeat for at most ten times
        #If the potential is entirely below 0,
        if (MeetingPoints[0] == None) or (MeetingPoints[1] == None):
            print('Resetting the energy guess!')
            print(E_guess_try.values())
            smallerenergies = [k for j,k in E_guess_try.values() if k < E_guess]
            if len(smallerenergies)>0:
                if E_guess > max(PotentialArray):
                    #Eguess is too big and is unbound!
                    E_guess = (E_guess + max(smallerenergies))/2
            else:
                #no smaller energies? div by 2!
                print("No smaller energies???")
                import matplotlib.pyplot as plt
                #plt.plot(PotentialArray,label="potential array")
                #plt.plot(np.ones(len(PotentialArray))*E_guess)
                #plt.legend()
                #plt.show()
                if E_guess < min(PotentialArray):
                    #Eguess is too small and unphysical!
                    print("increasing E_guess")
                    E_guess = (max(PotentialArray) - min(PotentialArray))/2

            iteration += 1
            print('E_guess: ',E_guess)
            if iteration > 10:
                end_program = True
                break
        else:
            p = 0
            MeetingPoints = tuple(MeetingPoints)

    return MeetingPoints,end_program,E_guess


def DetermineMinAndMax(MeetingPoints):
    #Is this really relevant? Why not just set the max and min to a fixed -1 and 1?
    #This is especially important since we're gonna train a symbolic regressor and 
    #wildly fluctuating domains will probably cause all sorts of nonsense to happen...
    #Sets the min and max as the half of the distance between the min and the max plus the min or the max
    #Used in WavefunctionNumerov as the xmin and xmax that the numerov algorithm will be applied to. 
    #Position_min = MeetingPoints[0] - (MeetingPoints[1] - MeetingPoints[0])/1
    #Position_max =  MeetingPoints[1] + (MeetingPoints[1] - MeetingPoints[0])/1

    return -5,5


#######################################
# iii) Calculate the wave function
def WaveFunctionNumerov(potential, E_guess, ngrids, Initial_augmentation, xmin, xmax):

    #Initializing the wave function
    WaveFunction = []

    #Setting the divisions
    deltax = (xmax-xmin)/ngrids

    #Setting the first values of the wave function
    WaveFunction.append((float(xmin),0))
    WaveFunction.append((float(xmin+deltax), Initial_augmentation))

    #Defing an array and an index to use in the for loop
    index = 0
    PositionArray = np.arange(xmin, xmax, deltax)
    Vx = []
    #Calculating the wave function for other values
    for i in np.arange(xmin + (2 * deltax), xmax, deltax):
        #Evaluating the potential
        #TODO: reconfigure this to be less hacky and just grab
        #potentials from Vxar.
        #For V_i+1
        x = PositionArray[index+1]
        V_plus1 = eval(potential) #runs i**2
        #print(potential,V_plus1)
        #print(1/0)
        #For V_i
        x = PositionArray[index]
        V = eval(potential)

        #For V_i-1
        x = PositionArray[index-1]
        V_minus1 = eval(potential)

        #Setting the k**2 values ( where k**2 = (2m/HBar)*(E-V(x)) )
        k_2_plus1 = 2 * (E_guess - V_plus1)
        k_2 = 2 * (E_guess - V)
        k_2_minus1 = 2 * (E_guess - V_minus1)

        #Calculating the wave function
        psi = ((2 * (1 - (5/12) * (deltax**2) * (k_2)) * (WaveFunction[-1][1])) - (1 + (1/12) * (deltax**2) * k_2_minus1 ) * (WaveFunction[-2][1])) / (1 + (1/12) * (deltax**2) * k_2_plus1)

        #Saving the wave function and the x coordinate
        WaveFunction.append((i,psi))
        Vx.append(V)
        #Incrementing the index
        index += 1
    #plt.plot(WaveFunction,label=r"$\psi(x)$")
    #plt.plot(Vx,label=r"$V(x)$")
    #plt.legend()
    #plt.show()
    print("call to numerov done")
    return WaveFunction


########################################################
# iv) Determine the number of nodes in the wave function

def NumberNodes(WaveFunction):
    #This function evaluates the number of nodes in the wavefunction. The number of nodes will allow us the determine the energy level to which a certain wave function corresponds.

    #Parameter:
    #----------
    #    WaveFunction (list) : Defines the wave function. Has the general form: [(x0, psi(x0)), (x1, psi(x1)), ...]

#    Returns:
#    --------
#        NumerberOfNodes (int) : Defines the number of nodes in the wave function (the number of time this function passed by the x axis). The number of nodes in a wave funtion
#                                corresponds to the energy level of that wave function
#        PositionNodes (list) : Defines the x position of all the nodes. Has the form : [position_nodes_1, position_nodes_2, ...]
#        x_max (float) : the greatest position of a node. Corresponds to the maximum value of PositionNodes
#


    #Initialize the number of nodes and their position
    NumberOfNodes = 0
    PositionNodes = list()

    #Calculate the number of nodes
    for i in range(1,len(WaveFunction)-1):
        if (WaveFunction[i][1] > 0 and WaveFunction[i+1][1] < 0) or (WaveFunction[i][1] < 0 and WaveFunction[i+1][1] > 0) or (WaveFunction[i][1] == 0):
            NumberOfNodes += 1
            PositionNodes.append(WaveFunction[i][0])


    #Gets the biggest position
    x = list()
    for position,wave in WaveFunction:
        x.append(position)
    x_max = max(x)

    return NumberOfNodes,PositionNodes,x_max


#####################################################
# v) Verify if wavefunction respects the required conditions for a well-behaved wavefunction

def VerifyTolerance(WaveFunction, Tolerance, E_guess, E_guess_try, NumberOfNodes):
    #See if the wave function for the given energy level respects the tolerance. The tolerance is defined in the parameters of the Numerov.py script. The tolerance is respected
    #if the last value of the wave function is smaller than this tolerance or if two energy guess are very very close. The function return yes in this case
    #and false otherwise.

    #Parameter:
    #----------
    #    WaveFunction (list) : Defines the wave function. Has the general form: [(x0, psi(x0)), (x1, psi(x1)), ...]
    #    Tolerance (float) : Defines the tolerance wich the wave function must respect
    #    E_guess (float) : The minimum value of the position for the potential
    #    E_guess_try (Dict) : a dictionnary that contains the previous energy guess. Has the form : {nbr_nodes1:[E_min,E_max], nbr_nodes2:[E_min,E_max],...}
    #    NumerberOfNodes (int) : Defines the number of nodes in the wave function (the number of time this function passed by the x axis). The number of nodes in a wave funtion
    #                            corresponds to the energy level of that wave function
#
#    Returns:
#   --------
#      VerificationTolerance (bool) : defines if the wave function respects the condition. Has the value 'yes' if it resects them and 'no' otherwise
#
    print(E_guess_try)
    VerificationTolerance = False
    # i) Checks if the last value of the wave function respects the tolerance
    print('\psi blowing up? $\psi(N) = ', WaveFunction[-1][1])

    # ii) Checks if the energy guess doesn't change a lot
    try:
        E_minus = E_guess_try[NumberOfNodes][1]
        E_plus = E_guess_try[NumberOfNodes + 1][0]
    except KeyError:
        print("No Energies to compare with yet...")
        return False
    if (E_guess < E_plus and E_guess > E_minus) and ((E_minus/E_plus) > 0.9999999999) and (np.abs(WaveFunction[-1][1])<Tolerance):
        VerificationTolerance = True            
    return VerificationTolerance

def CorrectNodeNumber(NumberOfNodes, PositionNodes, x_max, E_guess, E_guess_try):


    NumberOfNodesCorrected = NumberOfNodes
    #Correct the number of nodes if E_guess is between the lowest energy for this number of nodes and the maximum for the number of nodes - 1
    try:
        if (E_guess_try[NumberOfNodes][1] > E_guess) and (E_guess_try[NumberOfNodes - 1][1] < E_guess):
            NumberOfNodesCorrected -= 1
    #If the dictionnary E_guess_try doesn't contain these keys check if the Last number of nodes is close to the maximum value in x x_max
    except KeyError:
        if (PositionNodes/x_max) > 94:
            NumberOfNodesCorrected -= 1

    return NumberOfNodesCorrected


#######################################################
# vi) Saves energy and the correponding number of nodes

def SaveEnergy(NumberOfNodes, E_guess, E_guess_try):

    #Checks if the key Number of Nodes exists. If it doesn't, define the two values in the list corresponding to the key NumberOfNodes as E_guess.
    try:
        E_guess_try[NumberOfNodes]

    except KeyError:
        E_guess_try[NumberOfNodes] = [E_guess, E_guess]
        return E_guess_try

    #Checks if the energy guess is smaller than the smallest value in the list
    if E_guess < E_guess_try[NumberOfNodes][0]:
        E_guess_try[NumberOfNodes][0] = E_guess

    #Checks if the energy guess is greater than the biggest value in the list
    elif E_guess > E_guess_try[NumberOfNodes][1]:
        E_guess_try[NumberOfNodes][1] = E_guess

    return E_guess_try

  