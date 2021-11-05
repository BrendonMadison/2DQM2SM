#Author: Brendon Madison of University of Kansas Physics Department
#Date: Nov. 3rd-5th 2021
#Purpose: Solve an arbitrary two state (i.e. SU(2)) Hamiltonian for quantum mechanics (QM) purposes as well as statistical mechanics (SM) purposes.
#   QM Values Calculated:
#                    :Hamiltonian (H) , Time Propagator (U) , 
#                    :Time dependent wavefunction (psit) , Density matrix (rho),
#                    :SU(2) Projection Matrices (piplus etc.) , Probability of measuring said projections (probplus etc.)
#                    :Constructing density matrix for A and B states, Probability of measure A and B states of rho
#
#   SM Values Calculated:
#                    :Partition Function (Z,partfun) , Mean Energy (<E>,meanE) , 
#                    :Isochoric Heat Capacity (cV) , Entroypy (S,entS),
#                    :Chemical Potential (mu,chem) , Pressure (P,press)
#
#Secondary Purposes: To teach the following to PHSX 719 (Graduate Problem Solving)
#                    as well as a way to teach using sympy, github and, making your code easy to use and easy to learn.
#                    To test the neutron-antineutron oscillation values discussed in https://arxiv.org/pdf/0902.0834.pdf
#
#Code Flow:
#         :Look for argparse config filename->Import Config->Implement Settings->Construct Hamiltonian
#         :->Solve TDSE->Construct Density Matrix->Do measurements using projection measurements
#         :Code flow then stops and switches to statistical mechanics
#         :Use Hamiltonian to solve Partition Function->Solve for ln(partition function)
#         :->Use ln(Z) to solve <E> -> Use ln(Z) to solve c_V
#         :->Solve for Entropy -> Step back and construct partition function with N states
#         :->Repeat above for N states -> Solve for chemical potential and pressure
#         :->If plots are set to be saved then save plots
#
#Known issues:
#         :sympy truncates any matrix that has a zero element so you can't initialize with 0 values
#         :Small values of the "ai" parameters round to 0 which, expectedly, creates issues. Namely the neutron oscillation calculations dont work
#         :The chemical potential and pressure matrices are usually flat. Not necessarily an issue but its why those plots are flat
#         :sympy's simplification creates some weird errors in the QM and SM computations
#         :There are various error codes when dealing with complex values that should be suppressed

from sympy import symbols, exp, simplify, Matrix, diff, Trace, ln, log, cosh, tanh, plot, lambdify
from sympy.plotting import plot3d
from sympy.physics.quantum.dagger import Dagger

from scipy import integrate

#To handle the Fermi-Diract distribution computational errors
#Although, since this ins't support by sympy these errors likely still exist!
#This makes me both angry and sad
from scipy.special import expit
#So instead I will try to use the equality:         1/(exp(-ax)+1) = 1/2 * (1-tanh(-ax/2))
#as well as the equality:                           1/(exp(ax)+1) = 1/2 * (1-tanh(ax/2))

import matplotlib.pyplot as plt

import numpy as np

import configparser

#Import the ConfigParser function
from ConfigParser2D import ImportConfig

#Import os to create a directory
import os

#Use config parser to import dictionaries with settings, values
setd,qmv,smv,ptv = ImportConfig('NeutronOsc.ini')

try:
    os.mkdir(setd["name"])
    print("Created directory with name: ",setd["name"])
except:
    print("Directory of following name already exists: ",setd["name"])

#Define the symbols for the 2D Hamiltonian

a0,a1,a2,a3 = symbols('a0 a1 a2 a3')
#We do t and h separate so we can specify that they have to be real and positive
t = symbols('t',real=True,positive=True)
h = symbols('h',real=True,positive=True)
H = 0.5*Matrix([[a0+a3,a1-1j*a2],[a1+1j*a2,a0-a3]])

#since you don't need to specify a type in python you can load variables symbolically or numerically like so
#symbolically
#a0val = "a0"
#a1val = "a1"
#numerically
#a0val = 2
#a1val = 3
#This is then how you change specific values in the Hamiltonian for one
#H = H.subs(a0,a0val)
#For more than one
#H = H.subs({a0:a0val,a1:a1val})
#print(H)
#Then for the complex conjugate
#print(Dagger(H))

#Time propagator
U = Matrix([[exp(-1j*(a0+a3)*t/(2*h)),exp(-1j*(a1-1j*a2)*t/(2*h))],[exp(-1j*(a1+1j*a2)*t/(2*h)),exp(-1j*(a0-a3)*t/(2*h))]])

#Initial State
cA, cB = symbols('cA cB')
psi0 = Matrix([[cA],[cB]])
#If you give these 0 they will truncate the matrix so you have to give them a small value
psi0A = Matrix([[1],[1e-40]])
psi0B = Matrix([[1e-40],[1]])

#Time Dependent Wave Function
psit = U * psi0
psitA = U * psi0A
psitB = U * psi0B

#Probabilities of the cA*cA , cA*cB, cB*cA, cB*cB states
#aka the cA->cA , cA->cB , cB->cA , cB->cB probabilities
probaa = simplify(Dagger(psit)[0] * psit[0])
probab = simplify(Dagger(psit)[1] * psit[0])
probba = simplify(Dagger(psit)[0] * psit[1])
probbb = simplify(Dagger(psit)[1] * psit[1])

#normalize with respect to their possible transitions
probaa = probaa / (probaa+probab+probbb+probba)
probab = probab / (probaa+probab+probbb+probba)
probba = probba / (probbb+probba+probaa+probab)
probbb = probbb / (probbb+probba+probaa+probab)

#Density matrix from time dependent wave function
#Since there is only 1 possible wave function (of the two states) the weighting is 100%
rho = psit * Dagger(psit)
#We have the A and B density matrices to determine their probabilities later
rhoA = psitA * Dagger(psitA)
rhoB = psitB * Dagger(psitB)

#Shannon entropy of density matrix (information entropy)
entI = -1.0*np.trace(rho * ln(rho))

#Projection Operators for the + and - as well as the standard xyz basis

piplus = Matrix([[0,1],[0,0]])
piminus = Matrix([[0,0],[1,0]])
pix = 0.5*Matrix([[0,1],[1,0]])
piy = -0.5j*Matrix([[0,1],[-1,0]])
piz = 0.5*Matrix([[1,0],[0,-1]])

probplus = np.abs(np.trace(piplus * rho))
probminus = np.abs(np.trace(piminus * rho))
probx = np.abs(np.trace(pix * rho))
proby = np.abs(np.trace(piy * rho))
probz = np.abs(np.trace(piz * rho))

if setd["log"]==True:
    if setd["symbolic"]==True:
        print("Writing symbolic solution")
        fileqm = open(str(setd["name"])+"/QM.txt","w")
        fileqm.write(f'\nHamiltonian, H:\n{H}\n')
        fileqm.write(f'\nTime propagator, U:\n{U}\n')
        fileqm.write(f'\nTime Dependent Waveform, psi(t):\n{psit}\n')
        fileqm.write(f'\n---Assorted Wavefunction Mixing Probabilities---\n\nProb(A->A):\n{probaa}\nProb(A->B):\n{probab}\nProb(B->A):\n{probba}\nProb(B->B):\n{probbb}\n')
        fileqm.write(f'\nDensity Matrix:\n{rho}\n')
        fileqm.write(f'\nShannon Entropy of Density Matrix:\n{entI}\n')
        fileqm.write(f'\nEigenvalues (first value is eigenvalue second is multiplicity/degeneracy):\n{simplify(rho.eigenvals())}\n')
        #fileqm.write(f'\nEigenvectors:\n{rho.eigenvects()}\n')
        fileqm.write("\n----------\n\n---Probability of various spin projections---\n\n----------\n")
        fileqm.write(f'\nProbability of Plus:\n{probplus}\n')
        fileqm.write(f'\nProbability of Minus:\n{probminus}\n')
        fileqm.write(f'\nProbability of X:\n{probx}\n')
        fileqm.write(f'\nProbability of Y:\n{proby}\n')
        fileqm.write(f'\nProbability of Z:\n{probz}\n')
        fileqm.close()
    else:
        print("Writing numeric solution")
        
        fileqm = open(str(setd["name"])+"/QM.txt","w")
        
        #Rederive some values with the explicit forms. Doing this because sympy may simplify it further
        H = H.subs({a0:qmv["a0"],a1:qmv["a1"],a2:qmv["a2"],a3:qmv["a3"]})
        U = U.subs({a0:qmv["a0"],a1:qmv["a1"],a2:qmv["a2"],a3:qmv["a3"],h:qmv["h"]})
        #psi0.subs({cA:qmv["A"],cB:qmv["B"]})
        #The elements of a sympy matrix are in an array. Topleft is 0, topright is 1, bottom left is 2, bottom right is 3
        #print(f'U:{U[0]}\n{U[1]}\n{U[2]}\n{U[3]}')
        psit = U * psi0
        #Have to do the next line because, for some reason, cA and cB still exists in psit despite being substituted above
        psit = psit.subs({cA:qmv["A"],cB:qmv["B"]})
        
        #We can also do the density matrix and do measurements and plots for it
        rho = rho.subs({a0:qmv["a0"],a1:qmv["a1"],a2:qmv["a2"],a3:qmv["a3"],h:qmv["h"],cA:qmv["A"],cB:qmv["B"]})
        #Density matrices for the A and B states
        rhoA = rhoA.subs({a0:qmv["a0"],a1:qmv["a1"],a2:qmv["a2"],a3:qmv["a3"],h:qmv["h"],cA:qmv["A"],cB:qmv["B"]})
        rhoB = rhoB.subs({a0:qmv["a0"],a1:qmv["a1"],a2:qmv["a2"],a3:qmv["a3"],h:qmv["h"],cA:qmv["A"],cB:qmv["B"]})
        
        #Shannon entropy
        entI = entI.subs({a0:qmv["a0"],a1:qmv["a1"],a2:qmv["a2"],a3:qmv["a3"],h:qmv["h"],cA:qmv["A"],cB:qmv["B"]})
        
        fileqm.write(f'\nHamiltonian, H:\n{H}\n')
        fileqm.write(f'\nTime propagator, U:\n{U}\n')
        fileqm.write(f'\nTime Dependent Waveform, psi(t):\n{psit}\n')
        fileqm.write(f'\n---Assorted Wavefunction Mixing Probabilities---\n\nProb(A->A):\n{U[0]}\nProb(A->B):\n{U[1]}\nProb(B->A):\n{U[2]}\nProb(B->B):\n{U[3]}\n')
        fileqm.write(f'\nDensity Matrix:\n{rho}\n')
        fileqm.write(f'\nShannon Entropy of Density Matrix:\n{entI}\n')
        fileqm.write(f'\nEigenvalues (first value is eigenvalue second is multiplicity/degeneracy):\n{simplify(rho.eigenvals())}\n')
        #fileqm.write(f'\nEigenvectors:\n{rho.eigenvects()}\n')
        fileqm.write("\n----------\n\n---Probability of various spin projections---\n\n----------\n")

        #recalculate probabilities to have explicit value
        probplus = np.trace(piplus * rho)
        probminus = np.trace(piminus * rho)
        probx = np.trace(pix * rho)
        proby = np.trace(piy * rho)
        probz = np.trace(piz * rho)
        probA = np.trace(rhoA * rho)
        probB = np.trace(rhoB * rho)
        
        fileqm.write(f'\nProbability of Plus:\n{probplus}\n')
        fileqm.write(f'\nProbability of Minus:\n{probminus}\n')
        fileqm.write(f'\nProbability of X:\n{probx}\n')
        fileqm.write(f'\nProbability of Y:\n{proby}\n')
        fileqm.write(f'\nProbability of Z:\n{probz}\n')
        fileqm.write(f'\nProbability of state A:\n{probA}\n')
        fileqm.write(f'\nProbability of state B:\n{probB}\n')
        fileqm.close()
        
        #Check to see if plotting is enabled
        if setd["plots"] == True:
            print(f'Plotting enabled. Plotting and saving QM plots under {setd["name"]}.')
            #Redo the mixing/transition probabilities
            #These are the "unobservable" probabilities because they don't exist in a measurable way
            #They are just elements in the time propagator
            probaa = U[0]
            #probab = 1-probaa
            probbb = U[3]
            #probba = 1-probbb
            #This is the only observable probability, unless we include other waveforms.
            #Its psi(t) projected onto psi(t)
            probo = simplify(Dagger(psit) * psit)
            
            #Sympy gives matrices as different formats than arrays. This forces it to 1 element.
            probo = probo[0]
            #print("Transition probability from psi*psi",probo)
            
            #Since we have numerical solutions we can plot and save the plots
            
            tn = np.linspace(0,2.0*ptv["time"],2*ptv["tp"])
            #for neutron-antineutron oscillation tn = np.linspace(0,1e12,20000)
            f_aa = lambdify(t,probaa,'numpy')
            #f_ab = lambdify(t,probab,'numpy')
            #f_ba = lambdify(t,probba,'numpy')
            f_bb = lambdify(t,probbb,'numpy')
            f_00 = lambdify(t,probo,'numpy')
            
            #Get the integrand and cumulative distribution (integral) for the probability
            f_0tn = f_00(tn)
            f_0tnr = f_0tn.real
            f_0tnr = f_0tnr/np.amax(f_0tnr)
            f_0tnc = f_0tn.imag
            f_0tnc = f_0tnc/np.amax(f_0tnc)
            f_0tni = integrate.cumtrapz(f_0tn,tn,initial=0)
    #        f_0tn = f_0tn-np.amin(f_0tn)
    #        f_0tn = f_0tn/np.amax(f_0tn)
            
            #Plot the integral probability components
            plt.plot(tn[ptv["tp"]:2*ptv["tp"]],(f_0tni/np.amax(f_0tni))[ptv["tp"]:2*ptv["tp"]],label='Integral (Probability)')
            plt.plot(tn[ptv["tp"]:2*ptv["tp"]],f_0tnr[ptv["tp"]:2*ptv["tp"]],label='Integrand Real (Density)')
            plt.plot(tn[ptv["tp"]:2*ptv["tp"]],f_0tnc[ptv["tp"]:2*ptv["tp"]],label='Integrand Imaginary (Density)')
            plt.title("Wavefunction Normalized Density, Probability")
            plt.xlabel("Time (sec)")
            plt.ylabel("Probability or Density (normalized units)")
            plt.legend()
            plt.savefig(str(setd["name"])+'/IntegratedTransitionProb.png')
            plt.close()
            
            #Now that we have the time dependent we can get a power spectrum
            #This is particularly useful as it helps us see what frequencies or periods are at play
            #Creates the frequency axis
            freq = np.fft.rfftfreq((tn[ptv["tp"]:2*ptv["tp"]]).size,tn[1]-tn[0])
            #This inverts it (Period = 1/frequency) to get the period axis
            #Note that, by doing this, the uniform binning/spacing of is lost. So the bins are larger at higher values of period.
            per = np.log(1/(freq+0.1))/np.log(10)
            plt.plot(per,np.abs(np.fft.rfft((f_0tnr[ptv["tp"]:2*ptv["tp"]]))),label='Integrand Real Power Spectra')
            plt.plot(per,np.abs(np.fft.rfft((f_0tnc[ptv["tp"]:2*ptv["tp"]]))),label='Integrand Imaginary Power Spectra')
            plt.xlabel("log_10(Period) , log_10(sec)")
            plt.ylabel("Power (arbitrary units)")
            plt.title("Probability Density Power Spectrum")
            plt.legend()
            plt.savefig(str(setd["name"])+'/TransitionProbPowSpec.png')
            plt.close()
            
            #Plots and saves the probability "components"
            #normalize the probabilities
            f_aan = f_aa(tn)*f_aa(tn)
            f_aan = f_aan - np.amin(f_aan)
            f_aan = f_aan/np.amax(f_aan)
            f_ab = 1-f_aan

            f_bbn = f_bb(tn)*f_bb(tn)
            f_bbn = f_bbn - np.amin(f_bbn)
            f_bbn = f_bbn/np.amax(f_bbn)
            f_ba = 1-f_bbn
            
            #Plot the "unobservable" probabilities
            fig = plt.figure(figsize=(8,4),dpi=100)
            gs = fig.add_gridspec(1,2,hspace=0,wspace=0.5)
            axs = gs.subplots()
            axs[0].plot(tn[ptv["tp"]:2*ptv["tp"]],f_aan[ptv["tp"]:2*ptv["tp"]],label='A->A')
            axs[0].plot(tn[ptv["tp"]:2*ptv["tp"]],f_ab[ptv["tp"]:2*ptv["tp"]],label='A->B')
            axs[0].plot(tn[ptv["tp"]:2*ptv["tp"]],f_aan[ptv["tp"]:2*ptv["tp"]]+f_ab[ptv["tp"]:2*ptv["tp"]],label='Total A')
            axs[0].set(xlabel='Time (sec)',ylabel='Probability')
            axs[0].legend()
            axs[1].plot(tn[ptv["tp"]:2*ptv["tp"]],f_bbn[ptv["tp"]:2*ptv["tp"]],label='B->B')
            axs[1].plot(tn[ptv["tp"]:2*ptv["tp"]],f_ba[ptv["tp"]:2*ptv["tp"]],label='B->A')
            axs[1].plot(tn[ptv["tp"]:2*ptv["tp"]],f_ba[ptv["tp"]:2*ptv["tp"]]+f_bbn[ptv["tp"]:2*ptv["tp"]],label='Total B')
            axs[1].set(xlabel='Time (sec)',ylabel='Probability')
            fig.suptitle("Probabilities for states A and B")
            plt.legend()
            plt.xlabel("Time (sec)")
            plt.ylabel("Probability")
            plt.savefig(str(setd["name"])+'/WaveformProbs.png')
            plt.close()
            
            #The relative abundance of each state can be estimated using the previous "unobservable" probabilities
            #These relative abundances are observable, unlike the probabilities they consist of.
            plt.plot(tn[ptv["tp"]:2*ptv["tp"]],0.5*(f_ab[ptv["tp"]:2*ptv["tp"]]+f_bbn[ptv["tp"]:2*ptv["tp"]]),label='Relative # B')
            plt.plot(tn[ptv["tp"]:2*ptv["tp"]],0.5*(f_ba[ptv["tp"]:2*ptv["tp"]]+f_aan[ptv["tp"]:2*ptv["tp"]]),label='Relative # A')
            plt.title("Relative Abundance of A and B")
            plt.xlabel("Time (sec)")
            plt.ylabel("Relative Counts (Normalized #)")
            plt.legend()
            plt.savefig(str(setd["name"])+'/RelativeAbundance.png')
            plt.close()
            
            #Normalize using these 5 as a basis. Not a good idea in terms of physics but it makes the plot presentable.
            pplus = probplus
            pminus = probminus
            ppx = probx
            ppy = proby
            ppz = probz
            ppA = probA
            ppB = probB
            
            #lambdify the probabilities for plotting
            f_pls = lambdify(t,pplus,'numpy')
            f_min = lambdify(t,pminus,'numpy')
            f_x = lambdify(t,ppx,'numpy')
            f_y = lambdify(t,ppy,'numpy')
            f_z = lambdify(t,ppz,'numpy')
            f_A = lambdify(t,ppA,'numpy')
            f_B = lambdify(t,ppB,'numpy')
            
            #Plot the x y z basis
            fig = plt.figure(figsize=(8,4),dpi=200)
            plt.plot(tn,np.abs(f_x(tn)),label='Prob. x')
            plt.plot(tn,np.abs(f_y(tn)),label='Prob. y')
            plt.plot(tn,np.abs(f_z(tn)),label='Prob. z')
            plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
            plt.tight_layout(pad=2)
            plt.xlabel('Time (sec)')
            plt.ylabel('Probability')
            plt.title("Probabilities for x y z Projections on Density Matrix ")
            plt.savefig(str(setd["name"])+'/ProjXYZProbs.png')
            plt.close()
            
            #Plot the + - basis
            fig = plt.figure(figsize=(8,4),dpi=200)
            plt.plot(tn,np.abs(f_pls(tn)),label='Prob. +')
            plt.plot(tn,np.abs(f_min(tn)),label='Prob. -')
            plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
            plt.tight_layout(pad=2)
            plt.xlabel('Time (sec)')
            plt.ylabel('Probability')
            plt.title("Probabilities for +- Projections on Density Matrix ")
            plt.savefig(str(setd["name"])+'/ProjPlusMinusProbs.png')
            plt.close()            
            
            #plot the A and B probabilities
            #since the A and B density matrices aren't unitary we have to normalize them here
            fig = plt.figure(figsize=(8,4),dpi=200)
            plt.plot(tn,np.abs(f_A(tn))/np.amax(np.abs(f_A(tn))),label='Prob. A')
            plt.plot(tn,np.abs(f_B(tn))/np.amax(np.abs(f_B(tn))),label='Prob. B')
            plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
            plt.tight_layout(pad=2)
            plt.xlabel('Time (sec)')
            plt.ylabel('Probability')
            plt.title("Probabilities for states A and B on Density Matrix ")
            plt.savefig(str(setd["name"])+'/ABProbs.png')
            plt.close()
            
            #plot the Shannon entropy
            f_entI = lambdify(t,entI,'numpy')
            fig = plt.figure(figsize=(8,4),dpi=200)
            plt.plot(tn,(f_entI(tn))[0][0],label='Shannon Entropy')
            plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
            plt.tight_layout(pad=2)
            plt.xlabel('Time (sec)')
            plt.ylabel('Entropy (unitless)')
            plt.title("Shannon/Information Entropy of Density Matrix")
            plt.savefig(str(setd["name"])+'/EntropyQM.png')
            plt.close()
        else:
            print("Plotting not enabled.")

#Now we switch to the statmech part of the calculations
#
#Why do this? Well, from an experimental point of view you don't use "raw" or "singular" neutrons
#You use ensembles of usually millions or billions of particles and they interact with eachother
#and their media as they transmit BEFORE any quantum interaction happens.
#
#So, if your statmech interactions/processes degrade your particles beyond the point of where
#they can be used for your experiment then you may not measure anything to statistical significance.
#Even if your initial state particles should be able to, the statmech noise dominates.
#
#symbols:
#beta symbol = 1/kB*T is thermal energy
#kB is boltzmann constant
#N is the number of these identical states
#r is the number density
#V is the volume
beta = symbols("beta",real=True,positive=True)
N = symbols("N",real=True,positive=True)
kB = symbols("kB",real=True,positive=True)
V = symbols("V",real=True,positive=True)
r = symbols("r",real=True,positive=True)

#This is here for calculation purposes. Partition function.
partfun = simplify(np.trace(Matrix([[exp(-0.5*beta*(a0+a3)),exp(-0.5*beta*(a1-1j*a2))],[exp(-0.5*beta*(a1+1j*a2)),exp(-0.5*beta*(a0-a3))]])))

#From here on the simplification may be done by hand because, sadly, sympy's simplification is not good for this.
#Though this is checked using the setting in the parser "simp" in the parser or "simplification" in the .ini
#Particularly it is bad at recognizing it can use the ln(x*y) = ln(x) + ln(y) relationship to simplify further

#Checking again...hand simplification and machine simplification actually give different values in some places
#From my own checks, this is because sympy incorrectly calculates or simplifies some things in these equations
#Most notably it messes up the entropy and heat capacity calculations
#So you are warned... the machine simplification does allow you to use any form though
if setd["simp"] == True:
    print("Hand simplification used for statistical mechanics results.")
    #Calculate 1 particle characteristics
    #natural log partition function, ln(Z)
    lnZ = -0.5*beta*(a0-a3) + ln(exp(-1*a3*beta) + 1)
    #Mean energy , <E>. Includes thermal and quantum parts
    #meanE = 0.5*(a0-a3) + a0/(exp(beta*a0)+1)
    #meanE = 0.5*(a0-a3) + a0*expit(-1.0*beta*a0)
    meanE = 0.5*(a0-a3) + a3*0.5*(1.0-tanh(beta*a3))
    #Thermal entropy. Not the Shannon (aka cannonical or information) entropy.
    #entS = kB * (ln(exp(-beta*a0)+1) - beta*a0/(exp(beta*a0)+1))
    #entS = kB * (ln(1.0/expit(beta*a0)) - beta*a0*expit(-1.0*beta*a0))
    entS = kB * (ln(exp(-beta*a3)+1) - beta*a3*0.5*(1.0-tanh(beta*a3)))
    #Isochoric heat capacity, i.e. heat capacity for a fixed volume
    #cV = a0**2 * beta**2 * kB * 1/(2*cosh(a0*beta) + 2)
    #cV = a0**2 * beta**2 * kB * exp(beta*a0)/(exp(beta*a0) + 1)
    cV = a3**2 * beta**2 * kB * (1 - 0.5*(1.0-tanh(beta*a3)))
    #cV = a0**2 * beta**2 * kB * (1 - expit(-1.0 * beta*a0))
    
    #Somewhat repeating now but for a system with N particles
    #How do we do this? Well we assume they all share the same Hamiltonian and then sum them such that
    #H' = sum_i^N H = N * H
    #Thus, Z_N = tr(exp(-beta*H')) = tr(exp(-beta*N*H))
    #So your partition function picks up an extra exp(N) factor and the ln(Z) picks up an extra N
    #This also assumes they are identical and non-interacting.
    #Else you pick up 1/N! and other things
    #Our new N dependent values:
    lnZ_N = -0.5*beta*N*(a0-a3) + ln(exp(-1*a3*beta*N) + 1)
    #meanE_N = 0.5*N*(a0-a3) + a0*N/(exp(beta*a0*N)+1)
    meanE_N = 0.5*N*(a0-a3) + a3*N*0.5*(1.0-tanh(beta*a3*N))
    #entS_N = kB * (ln(exp(-beta*a0*N)+1) - beta*a0*N/(exp(beta*a0*N)+1))
    entS_N = kB * (ln(exp(-beta*a3*N)+1) - beta*a3*N*0.5*(1.0-tanh(beta*a3*N)))
    #cV_N = a0**2 * N**2 * beta**2 * kB * (1 - 1/(exp(beta*a0*N)+1))
    cV_N = a3**2 * N**2 * beta**2 * kB * (1 - 0.5*(1.0-tanh(beta*a3*N)))
    #Since we have number we can get the chemical potential by differentiating with respect to N
    #chem = -1.0/beta * diff(lnZ_N,N)
    #chem = -1.0/beta*( (beta*(a0-a3) - a0*beta/(exp(a0*beta*N)+1))
    #chem = (a0/(exp(a0*beta*N)+1) - (a0-a3))
    chem = (a3*0.5*(1.0-tanh(beta*a3*N)) - (a0-a3))
    #Remember that N = n * V where n is the number density and V is the volume.
    #We are treating N as a variable and n as a variable to keep V constant since we used the isochoric heat capacity.
    #In other words, we can transform from N to V by using this relationship V = N/n ...
    #Then we can differentiate with respect to volume in order to get the pressure.
    lnZ_V = -0.5*beta*r*V*(a0-a3) + ln(exp(-1*a3*beta*r*V) + 1)
    #press = 1/beta * diff(lnZ_V,V) -> this is similar to chemical potential
    #press = r*((a0-a3) - a0/(exp(a0*beta*r*V)+1)) #Check that the units are in Joules/Volume
    press = r*((a0-a3) - a3*0.5*(1.0-tanh(beta*a3*r*V)))
else:
    print("Machine simplification and solutions used for statistical mechanics results.")
    #Calculating 1 particle characteristics ... to see explanations and notes read the above section
    #natural log partition function, ln(Z)
    lnZ = ln(partfun)
    #Mean energy , <E>. Includes thermal and quantum parts
    meanE = simplify(-1*diff(lnZ,beta))
    #Thermal entropy. Not the Shannon (aka cannonical or information) entropy.
    entS = kB*lnZ + kB*beta*meanE
    #Isochoric heat capacity, i.e. heat capacity for a fixed volume
    cV = kB * beta**2 * diff(diff(lnZ,beta),beta)
    #Notice that cV and entropy have the same units of Energy/Temperature
    #Repeat for N dependent terms and partition function
    lnZ_N = -0.5*beta*N*(a0-a3) + ln(exp(-1*a3*beta*N) + 1)
    meanE_N = simplify(-1*diff(lnZ_N,beta))
    entS_N = kB*lnZ_N + kB*beta*meanE_N
    cV_N = kB * beta**2 * diff(diff(lnZ_N,beta),beta)
    #The N dependent terms, chemical potential and pressure
    chem = -1.0/beta * diff(lnZ_N,N)
    lnZ_V = -0.5*beta*r*V*(a0-a3) + ln(exp(-1*a3*beta*r*V) + 1)
    press = 1/beta * diff(lnZ_V,V)
    
if setd["symbolic"] == False:
    partfun = partfun.subs({a0:qmv["a0"],a3:qmv["a3"]})
    lnZ = lnZ.subs({a0:qmv["a0"],a3:qmv["a3"],kB:smv["kB"]})
    lnZ_N = lnZ_N.subs({a0:qmv["a0"],a3:qmv["a3"],kB:smv["kB"]})
    meanE = meanE.subs({a0:qmv["a0"],a3:qmv["a3"],kB:smv["kB"]})
    meanE_N = meanE_N.subs({a0:qmv["a0"],a3:qmv["a3"],kB:smv["kB"]})
    cV = cV.subs({a0:qmv["a0"],a3:qmv["a3"],kB:smv["kB"]})
    cV_N = cV_N.subs({a0:qmv["a0"],a3:qmv["a3"],kB:smv["kB"]})
    entS = entS.subs({a0:qmv["a0"],a3:qmv["a3"],kB:smv["kB"]})
    entS_N = entS_N.subs({a0:qmv["a0"],a3:qmv["a3"],kB:smv["kB"]})
    chem = chem.subs({a0:qmv["a0"],a3:qmv["a3"],kB:smv["kB"]})
    press = press.subs({a0:qmv["a0"],a3:qmv["a3"],r:(smv["N"]/smv["V"]),kB:smv["kB"]})
    if setd["plots"] == True:
        print("Creating statmech plots.")
        
        #create the two axes , beta and N. Volume will be derived from N
        bn = np.linspace(1.0/(smv["kB"] * ptv["Tmin"]),1.0/(smv["kB"] * ptv["Tmax"]),ptv["tp"])
        nn = np.linspace(ptv["Nmin"],ptv["Nmax"],ptv["tp"])
        #create meshgrid for 2d plots
        X,Y = np.meshgrid(bn,nn)
        
        #Lambdify , this turns all the symbolic forms into functions that can be evaluated
        f_me = lambdify(beta,meanE,'numpy')
        f_men = lambdify([beta,N],meanE_N,'numpy')
        f_cv = lambdify(beta,cV,'numpy')
        f_cvn = lambdify([beta,N],cV_N,'numpy')
        f_es = lambdify(beta,entS,'numpy')
        f_esn = lambdify([beta,N],entS_N,'numpy')
        f_ch = lambdify([beta,N],chem,'numpy')
        f_ps = lambdify([beta,V],press,'numpy')
        
        #Plot the 1D plots (the ones that depend only on beta)
        #These are plotting using temperature though
        fig = plt.figure(figsize=(12,4),dpi=100)
        gs = fig.add_gridspec(1,3,hspace=0,wspace=0.5)
        axs = gs.subplots()
        axs[0].plot(bn,f_me(bn),label='<E>')
        axs[0].set(xlabel='beta, [1/Joules]',ylabel='<E> , [Joules]',title='Mean Energy per Temperature')
        axs[0].legend()
        axs[1].plot(bn,f_cv(bn),label='c_V')
        axs[1].set(xlabel='beta, [1/Joules]',ylabel='c_V , [Joules/Kelvin]',title='Isochoric Heat Capacity')
        axs[1].legend()
        axs[2].plot(bn,f_es(bn),label='S')
        axs[2].set(xlabel='beta, [1/Joules]',ylabel='Entropy, [Joules/Kelvin]',title='Entropy per Temperature')
        axs[2].legend()
        fig.suptitle("")
        plt.savefig(str(setd["name"])+'/1ParticleSM.png')
        plt.close()
        
        #Plot the 2D plots (the ones that depend on beta and N)
        #Start with <E>, cV, S
        #These are plotting using temperature though
        fig = plt.figure(figsize=(12,4),dpi=100)
        gs = fig.add_gridspec(1,3,hspace=0,wspace=0.5)
        axs = gs.subplots()
        axs[0].pcolormesh(X,Y,f_men(X,Y),label='<E>_N',shading='auto')
        axs[0].set(xlabel='beta, [1/Joules]',ylabel='Number, N',title='Mean Energy per Temperature and N')
        #axs[0].legend()
        axs[1].pcolormesh(X,Y,f_cvn(X,Y),label='c_V_N',shading='auto')
        axs[1].set(xlabel='beta, [1/Joules]',ylabel='Number, N',title='Isochoric Heat Capacity')
        #axs[1].legend()
        axs[2].pcolormesh(X,Y,f_esn(X,Y),label='S_N',shading='auto')
        axs[2].set(xlabel='beta, [1/Joules]',ylabel='Number, N',title='Entropy per Temperature and N')
        #axs[2].legend()
        fig.suptitle("")
        plt.savefig(str(setd["name"])+'/NParticlesSM.png')
        plt.close()
        
        #Now with chemical potential and pressure
        fig = plt.figure(figsize=(8,4),dpi=100)
        gs = fig.add_gridspec(1,2,hspace=0,wspace=0.5)
        axs = gs.subplots()
        axs[0].pcolormesh(X,Y,f_ch(X,Y),label='mu',shading='auto')
        axs[0].set(xlabel='beta, [1/Joules]',ylabel='Number, N',title='Chemical Potential')
        #axs[0].legend()
        axs[1].pcolormesh(X,Y/smv["r"],f_ps(X,Y/smv["r"]),label='P_N',shading='auto')
        axs[1].set(xlabel='beta, [1/Joules]',ylabel='Volume, [meters^3]',title='Pressure')
        #axs[1].legend()
        fig.suptitle("")
        plt.savefig(str(setd["name"])+'/ChemPress.png')
        plt.close()
        

if setd["log"] == True:
    print("Generating statmech log.")
    filesm = open(str(setd["name"])+"/SM.txt","w")
    filesm.write(f'\nPartition function, Z:\n{partfun}\n')
    filesm.write(f'\nlog of partition function, lnZ:\n{lnZ}\n')
    filesm.write(f'\nMean Energy, <E>:\n{meanE}\n')
    filesm.write(f'\nIsochoric Heat Capacity, C_V:\n{cV}\n')
    filesm.write(f'\nThermal Entropy, S:\n{entS}\n')
    
    filesm.write("\n\n---Repeating for a system of N of these identical states---\n\n")
    
    filesm.write(f'\nlog of partition function, lnZ_N:\n{lnZ_N}\n')
    filesm.write(f'\nMean Energy, <E_N>:\n{meanE_N}\n')
    filesm.write(f'\nIsochoric Heat Capacity, C_VN:\n{cV_N}\n')
    filesm.write(f'\nThermal Entropy, S_N:\n{entS_N}\n')
    filesm.write(f'\nChemical potential, mu:\n{chem}\n')
    filesm.write(f'\nPressure, P_N:\n{press}\n')
    filesm.close()


