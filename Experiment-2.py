"""
Author: Brendan Patch

Date: 14 Feb 2020

Purpose: Experiment 1 in the paper 'Analyzing large frequency disruptions in power
systems using large deviations theory' by Brendan Patch and Bert Zwart

Notes: 
- As written it should run in Spyder. 
- Changing sigmaStep and the inputs to the linspace function used for arrival rate will improve resolution but slow down computation time
"""

from IPython import get_ipython
get_ipython().magic('reset -sf') 
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import math
import seaborn as sns
import matplotlib.pyplot as plt
#import tikzplotlib


numberFailuresMax = 2
rangeNumberFailures = range(numberFailuresMax+1)

epsilon = 0.1

rho = 1/30
alpha = 12.5
mu = 12
beta = 0.05
delta = 1
planningHorizon = 2.0
gamma = 0.05
numberFails = 1

sigmaMin = 0.1*np.sqrt(0.1)
sigmaMax = 1.5*np.sqrt(0.1)
sigmaStep = 0.1
rangeSigmaValues = np.arange(sigmaMin,sigmaMax,sigmaStep)

rangeArrivalsValues = []
for i in np.linspace(1,10,10):
    rangeArrivalsValues.insert(0,0.5/10**i)

A_deterministic = np.array([[0,1], [-beta/mu, -alpha/mu]])
b_deterministic = np.array([[0],[delta*numberFails/mu]])
C1_deterministic = np.concatenate((A_deterministic,b_deterministic),axis=1)
C2_deterministic = np.concatenate((np.zeros((1,2)),np.zeros((1,1))),axis=1)
C_deterministic = np.concatenate((C1_deterministic, C2_deterministic), axis=0)

tt = np.arange(0,planningHorizon+0.01,0.01)
oneFail_deterministic_sample_path = []
for time in tt:
    foo = -np.array([0,1,0]).dot(expm(C_deterministic*time))
    oneFail_deterministic_sample_path.append(foo[2])

numberFails = 2

b_deterministic = np.array([[0],[delta*numberFails/mu]])
C1_deterministic = np.concatenate((A_deterministic,b_deterministic),axis=1)
C2_deterministic = np.concatenate((np.zeros((1,2)),np.zeros((1,1))),axis=1)
C_deterministic = np.concatenate((C1_deterministic, C2_deterministic), axis=0)

twoFail_deterministic_sample_path = []
for time in tt:
    foo = -np.array([0,1,0]).dot(expm(C_deterministic*time))
    twoFail_deterministic_sample_path.append(foo[2])
    
gamma = -min(twoFail_deterministic_sample_path)
#print(gamma)

c1 = 0.0
c2 = 0.0
c3 = 0.0
t = planningHorizon/3.0

counter = 0

a1 = beta**2*rho**2/mu**2
a2 = (2*beta*rho**2*mu-beta**2-alpha**2*rho**2)/mu**2
a3 = (alpha**2+rho**2*mu**2-2*mu*beta)/mu**2

A = np.array([[0,1,0,0,0,0], #1
          [0,0,1,0,0,0], #2
          [0,0,0,1,0,0], #3
          [0,0,0,0,1,0], #4
          [0,0,0,0,0,1], #5
          [a1, 0, a2 , 0 ,a3, 0]]) #6

b1 = beta*rho
b2 = beta+rho*alpha
b3 = alpha+rho*mu
b4 = mu
    
B = np.array([[b1**2,b1*b2, b1*b3, b1*b4, 0, 0], 
           [b2*b1, b2**2, b2*b3, b2*b4, 0, 0], 
           [b3*b1, b3*b2, b3**2, b3*b4, 0, 0], 
           [b4*b1, b4*b2, b4*b3, b4**2, 0, 0], 
           [0,0,0,0,0,0], 
           [0,0,0,0,0,0]])

C1 = np.concatenate((-np.transpose(A), B), axis=1)
C2 = np.concatenate((np.zeros((6,6)),A), axis=1)

C = np.concatenate((C1, C2), axis=0)

F = expm(C*planningHorizon)
    
G1 = F[0:6,6:12]
G2 = F[6:12, 6:12]

BT = np.transpose(G2).dot(G1)

probApproxData = []
pStarData = []
rateFunData = []
optNumberFailsData = []
for sigma in rangeSigmaValues:
    tempRowData1 = []
    tempRowData2 = []
    tempRowData3 = []
    tempRowData4 = []
    for arrivalRate in rangeArrivalsValues:
        tempData = []
        tempData2 = []
        solutionType = []
        nadirTime = []
        for numberFailures in rangeNumberFailures:
            
            def objective(c):
                y0 = np.array([[0], [0], [-delta*numberFailures/mu],  [c[0]], [c[1]], [c[2]]])
                foo = (0.5*(np.transpose(y0).dot(BT.dot(y0)))/(sigma**2/epsilon))-numberFailures*(epsilon*np.log(arrivalRate))  
                return foo.item(0)
            
            def constraint(c):
                y0 = np.array([[0], [0], [-delta*numberFailures/mu], [c[0]], [c[1]], [c[2]]])
                foo = expm(A*planningHorizon).dot(y0)
                return -foo.item(1)-gamma
        
            con = {'type': 'eq', 'fun': constraint}
            
            cInitial = np.array([[c1], [c2], [c3]])
                  
            sol = minimize(objective, cInitial, constraints=con, method='SLSQP',options={'disp': False,})
            
            tempData2.append(sol.x)
            
            optVal1 = sol.fun
            y0 = np.array([[0], [0], [-delta*numberFailures/mu],  [sol.x[0]], [sol.x[1]], [sol.x[2]]])
            frequencyDeviationsTemp = []
            for time in tt:
                foo = expm(A*time).dot(y0)
                frequencyDeviationsTemp.append(foo.item(1))
            if (np.where(frequencyDeviationsTemp <= -gamma)[0]).size > 0:
                timeOfNadir = tt[np.where(frequencyDeviationsTemp <= -gamma)[0][0]]
            else: timeOfNadir = planningHorizon
            nadirTime.append(timeOfNadir)
            F_shift = expm(C*timeOfNadir)
                
            G1_shift = F[0:6,6:12]
            G2_shift = F[6:12, 6:12]
            BT_shift = np.transpose(G2_shift).dot(G1_shift)
            foo = (0.5*(np.transpose(y0).dot(BT_shift.dot(y0)))/(sigma**2/epsilon))-numberFailures*(epsilon*np.log(arrivalRate))  
            
            optVal2 = foo.item(0)
        
            if optVal1 < optVal2: 
                tempData.append(optVal1)
                solutionType.append(1)
            else: 
                tempData.append(optVal2)
                solutionType.append(2)
        
        optimalNumberFailures_JSNE = np.argmin(tempData)
        cOpt_JSNE = tempData2[optimalNumberFailures_JSNE]
        optVal_JSNE = tempData[optimalNumberFailures_JSNE]
        optSolType_JSNE = tempData[optimalNumberFailures_JSNE]
        timeOfNadir_JSNE = nadirTime[optimalNumberFailures_JSNE]
        

        
        optVal_JJ = -2*epsilon*np.log(arrivalRate)
        
        optVals = [optVal_JJ, optVal_JSNE]
        optNumberFails = [2, optimalNumberFailures_JSNE]
        
        optSolMethod = np.argmin(optVals)
        
        optNumberFails = optNumberFails[optSolMethod]
        
        if optSolMethod == 0:
            y0 = np.zeros((6,1))
            y0[2] = -2*delta/mu
        else:
            y0 = np.array([[0], [0], [-delta*optNumberFails/mu], [cOpt_JSNE[0]], [cOpt_JSNE[1]], [cOpt_JSNE[2]]])

        def renewable_power(time):
            if optSolMethod == 0:
                wind = 0
            else:
                foo = expm(A*time).dot(y0)
                wind = abs(mu*foo.item(2)+alpha*foo.item(1)+beta*foo.item(0)+delta*optNumberFails)
            return wind
        
        pStar = renewable_power(planningHorizon)
        
        tempRowData1.append(min(optVals))
        tempRowData2.append(pStar) # Append p_star here
        tempRowData3.append(math.exp(-min(optVals)/epsilon))
        tempRowData4.append(optNumberFails) # This is method used to find solution (JSNE vs JSFN vs JJ)
    probApproxData.insert(0,tempRowData3)
    pStarData.insert(0,tempRowData2)
    rateFunData.insert(0,tempRowData1)
    optNumberFailsData.insert(0,tempRowData4)
    print(int(100*sigma/sigmaMax),'%')
    
    
    
plt.figure(figsize = (5,4))
sns.heatmap(rateFunData, cmap="PiYG")
#tikzplotlib.save("Ex2_RateVals.tex")

fig = plt.figure()
sns.heatmap(optNumberFailsData,xticklabels = rangeArrivalsValues, yticklabels = rangeSigmaValues[::-1], square = True, cmap="PiYG")
