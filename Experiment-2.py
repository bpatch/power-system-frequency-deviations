"""
Author: Brendan Patch

Date: 22 Feb 2020

Purpose: Experiment 2 in the paper 'Analyzing large frequency disruptions in power
systems using large deviations theory' by Brendan Patch and Bert Zwart

Notes: 
- As written it should run in Spyder. 
"""

from IPython import get_ipython
get_ipython().magic('reset -sf') 
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math
import seaborn as sns
#import tikzplotlib  #Uncomment to allow nicer plots to be created



sigmaMin = 0.03
sigmaMax = 0.5
sigmaStep = 0.01
rangeSigmaValues = np.arange(sigmaMin,sigmaMax+sigmaStep,sigmaStep)

rangeArrivalsValues = np.array([])
for i in np.linspace(1,10,10):
    rangeArrivalsValues = np.append(rangeArrivalsValues, 1/10**i)

rangeArrivalsValues = rangeArrivalsValues[::-1]

# Set experiment parameters
rho = 1/30
alpha = 12.5
mu = 12 
beta = 0.05
delta = 1
planningHorizon = 2.0
#sigma = 0.2916
#arrivalRate = 1/10**3
epsilon=0.1

timeStep = 0.01
tt = np.arange(0,planningHorizon+timeStep, timeStep)

# Determining the nadir from failure without wind

numberFails = 1

A_deterministic = np.array([[0,1], [-beta/mu, -alpha/mu]])
b_deterministic = np.array([[0],[delta*numberFails/mu]])
C1_deterministic = np.concatenate((A_deterministic,b_deterministic),axis=1)
C2_deterministic = np.concatenate((np.zeros((1,2)),np.zeros((1,1))),axis=1)
C_deterministic = np.concatenate((C1_deterministic, C2_deterministic), axis=0)

oneFail_deterministic_sample_path = []
for time in tt:
    foo = -np.array([0,1,0]).dot(expm(C_deterministic*time))
    oneFail_deterministic_sample_path.append(foo[2])

# Determining the nadir from two failures without wind

numberFails = 2

b_deterministic = np.array([[0],[delta*numberFails/mu]])
C1_deterministic = np.concatenate((A_deterministic,b_deterministic),axis=1)
C2_deterministic = np.concatenate((np.zeros((1,2)),np.zeros((1,1))),axis=1)
C_deterministic = np.concatenate((C1_deterministic, C2_deterministic), axis=0)

twoFail_deterministic_sample_path = []
for time in tt:
    foo = -np.array([0,1,0]).dot(expm(C_deterministic*time))
    twoFail_deterministic_sample_path.append(foo[2])
    
# Set the nadir level corresponding to that which occurs from two failures without wind
    
gamma = -min(twoFail_deterministic_sample_path)


# Optimising over time of nadir and number of failures, with failures at time 0

maxK = 5



# Initial conditions for minimize function later
c1 = 1
c2 = 1
c3 = 1
cInitial = np.array([[c1], [c2], [c3]])


# Set up counters for progress indicator
counter1Max = tt.size
counter2Max = maxK
counter3Max = rangeSigmaValues.size
counter4Max = rangeArrivalsValues.size

# Arrays to store results in
pStarData = []
rateFunData = []
optNumberFailsData = []
optNadirTimeData = []
optFailTimeData = []



thresholdSigmas0 = []
thresholdSigmas1 = []

counter4 = 0
for arrivalRate in rangeArrivalsValues:
    pStarRowData = []
    rateFunRowData = []
    optNumberFailsRowData = []
    optNadirTimeRowData = []
    # Loop over different values of inertia
    counter3 = 0
    firstK1 = 0
    firstK0 = 0
    for sigma in rangeSigmaValues: #
    
        # Initialise optimal rate fun (to be made smaller over k and nadir time loops)
        optVal = math.inf
        # Optimise over k and nadir time using brute force:
        counter2 = 0
        for k in range(maxK):
            
            counter1 = 0
            for nadirTime in tt:
           
                h1 = beta*rho
                h2 = beta+rho*alpha
                h3 = alpha+rho*mu
                h4 = mu
                h8 = rho*delta*k
                    
                H = np.array([[h1**2,h1*h2, h1*h3, h1*h4, 0, 0, 0, h1*h8], 
                           [h2*h1, h2**2, h2*h3, h2*h4, 0, 0, 0, h2*h8], 
                           [h3*h1, h3*h2, h3**2, h3*h4, 0, 0, 0, h3*h8], 
                           [h4*h1, h4*h2, h4*h3, h4**2, 0, 0, 0, h4*h8], 
                           [0,0,0,0,0,0,0,0], 
                           [0,0,0,0,0,0,0,0], 
                           [0,0,0,0,0,0,0,0], 
                           [h8*h1, h8*h2, h8*h3, h8*h4, 0, 0, 0, h8*h8]]) 
                
                H1 = H[0:7,0:7] 
                
                H2 = H[0:7,-1].reshape(-1,1) 
                
                a1 = beta**2*rho**2/mu**2
                a2 = (2*beta*rho**2*mu-beta**2-alpha**2*rho**2)/mu**2 
                a3 = (alpha**2+rho**2*mu**2-2*mu*beta)/mu**2 
                
                A1 = np.array([a1, 0, a2, 0, a3, 0]) 
                
                A = np.hstack((np.zeros((7,1)), np.vstack((np.identity(6), A1)))) 
                
                dummyMatrix1 = expm(  np.vstack( (np.hstack((-np.transpose(A), H1))  , np.hstack((np.zeros((7,7)), A))) ) * planningHorizon )
                
                dummyMatrix2 = expm(  np.vstack( (np.hstack((0, np.transpose(H2)[0]))  , np.hstack((np.zeros((7,1)), A))) ) * planningHorizon )
                
                dummyMatrix3 = expm( np.vstack( (np.hstack((-np.transpose(A), H2))  , np.zeros((1,8)) )) * planningHorizon )
                
                B11 = dummyMatrix1[0:7, 7:]
                
                B12 = dummyMatrix1[7:,7:]
                
                B1 = np.dot(np.transpose(B12), B11)
                
                B2 = dummyMatrix2[0,1:]
                
                B31 = dummyMatrix3[:7,-1].reshape(-1,1)
                
                B3 = np.dot(np.transpose(B12), B31)
                
                def objective(c):
                    y0 = np.array([0, 0, -delta*k/mu,  c[0], c[1], c[2], -k*delta*a2/mu+a3*c[2]+beta*rho**2*delta*k/mu**2])
                    integralValue = 0.5*(np.linalg.multi_dot([np.transpose(y0), B1, y0]) +np.dot(B2, y0) + np.dot(np.transpose(y0),B3)+(rho*delta*k)**2)/(sigma**2/epsilon)  
                    return integralValue.item(0)-k*epsilon*np.log(arrivalRate)
                
                def constraint(c):
                    y0 = np.array([0, 0, -delta*k/mu,  c[0], c[1], c[2], -k*delta*a2/mu+a3*c[2]+beta*rho**2*delta*k/mu**2])
                    valueAtNadirTime = np.dot(expm(A*nadirTime),y0)
                    return -valueAtNadirTime.item(1)-gamma
                
                con = {'type': 'eq', 'fun': constraint}
                
                c1 = 1
                c2 = 1
                c3 = 1
                cInitial = np.array([[c1], [c2], [c3]])
                
                sol = minimize(objective, cInitial, constraints=con, method='SLSQP', options={'disp': False})
                
        #        print('k=', k, 'nadir time = ', nadirTime, 'opt rate fun val = ', sol.fun) # Uncomment this line to see rate function values for different k and nadir time combinations       
        
                if sol.fun < optVal:
                    optNadirTime = nadirTime
                    optK = k
                    optVal = sol.fun
                    optSolx = sol.x
                    optSol = sol
                    optInertia = mu
                counter1 += 1
                print(format(100*(counter1 + counter2*counter1Max + counter3*counter1Max*counter2Max + counter4*counter3Max*counter1Max*counter2Max)/(counter1Max*counter2Max*counter3Max*counter4Max), '7.2f'), '%')
            counter2 += 1
        if optK == 1 and firstK1 == 0:
            thresholdSigmas1.append(sigma)
            firstK1 = 1
        if optK == 0 and firstK0 == 0:
            thresholdSigmas0.append(sigma)
            firstK0 = 1
        cInitial = sol.x
        y0 = np.array([0, 0, - delta*optK/mu,  optSolx[0], optSolx[1], optSolx[2], -optK*delta*a2/mu+a3*optSolx[2]+beta*rho**2*delta*optK/mu**2])
        foo = np.dot(expm(A*planningHorizon), y0)
        pStarRowData.append(mu*foo.item(2)+alpha*foo.item(1)+beta*foo.item(0)+delta*optK)
        rateFunRowData.append(optVal)
        optNumberFailsRowData.append(optK)
        optNadirTimeRowData.append(optNadirTime)
        counter3 += 1
    if counter4 == 0:
        pStarData = np.array(pStarRowData)[::-1].reshape(-1,1)
        rateFunData = np.array(rateFunRowData)[::-1].reshape(-1,1)
        optNumberFailsData = np.array(optNumberFailsRowData)[::-1].reshape(-1,1)
        optNadirTimeData = np.array(optNadirTimeRowData)[::-1].reshape(-1,1)
    else:
        pStarData = np.hstack((np.array(pStarData), np.array(pStarRowData)[::-1].reshape(-1,1)))
        rateFunData = np.hstack((np.array(rateFunData), np.array(rateFunRowData)[::-1].reshape(-1,1)))
        optNumberFailsData = np.hstack((np.array(optNumberFailsData), np.array(optNumberFailsRowData)[::-1].reshape(-1,1)))
        optNadirTimeData = np.hstack((np.array(optNadirTimeData), np.array(optNadirTimeRowData)[::-1].reshape(-1,1)))
    counter4 += 1

plt.figure(figsize = (5,4))
sns.heatmap(rateFunData,xticklabels = rangeArrivalsValues, yticklabels = rangeSigmaValues[::-1],cmap="PiYG")  
#tikzplotlib.save("Ex2_RateVals_22Feb.tex")
#ax.set_title('J')

#ax = plt.axes()
#plt.figure(figsize = (5,4))
#sns.heatmap(pStarData, xticklabels = rangeArrivalsValues, yticklabels = rangeSigmaValues[::-1], cmap="PiYG", ax = ax)  
#ax.set_title('p*')
#
#ax = plt.axes()
#plt.figure(figsize = (5,4))
#sns.heatmap(optNumberFailsData ,xticklabels = rangeArrivalsValues, yticklabels = rangeSigmaValues[::-1], cmap="PiYG", ax = ax)  
#tikzplotlib.save("Ex2_RateVals_22Jan.tex")

thresholdSigmas0 = np.hstack((np.array(thresholdSigmas0),sigmaMax*np.ones((1,len(rangeArrivalsValues)-len(thresholdSigmas0)))[0]))

plt.figure()
plt.plot(range(len(rangeArrivalsValues)),thresholdSigmas1,color="gray", alpha=0.3)
plt.plot(range(len(rangeArrivalsValues)),thresholdSigmas0,color="gray", alpha=0.6)
plt.gca().fill_between(range(len(rangeArrivalsValues)), sigmaMin*np.ones(len(rangeArrivalsValues)), thresholdSigmas1, alpha=0.3,color="gray")
plt.gca().fill_between(range(len(rangeArrivalsValues)), thresholdSigmas1, thresholdSigmas0, alpha=0.6,color="gray")
plt.gca().fill_between(range(len(rangeArrivalsValues)), thresholdSigmas0, sigmaMax*np.ones(len(rangeArrivalsValues)), alpha=0.9,color="gray")
plt.ylim(sigmaMin,sigmaMax)
plt.xlim(0,len(rangeArrivalsValues))
#tikzplotlib.save("Ex2_k_and p_star.tex")

#
#plt.figure(figsize = (5,4))
#sns.heatmap(optNadirTimeData, cmap="PiYG") 
 
