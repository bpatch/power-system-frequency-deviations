"""
Author: Brendan Patch

Date: 14 Feb 2020

Purpose: Experiment 3 in the paper 'Analyzing large frequency disruptions in power
systems using large deviations theory' by Brendan Patch and Bert Zwart

Notes: 
- As written it should run in Spyder. 
"""

from IPython import get_ipython
get_ipython().magic('reset -sf') 
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt
#import tikzplotlib

numberFailuresMax = 2
rangeNumberFailures = range(numberFailuresMax+1)

rho = 1/30
alpha = 12.5
mu = 12
beta = 0.05
delta = 1
planningHorizon = 2.0
gamma = 0.05
sigma = 0.29162277660168384
arrivalRate = 1/10**3
numberFails = 1
epsilon=0.1

inertiaMin = 2
inertiaMax = 20
inertiaStep = 0.1
rangeInertiaValues = np.arange(inertiaMin,inertiaMax,inertiaStep)

A_deterministic = np.array([[1/mu,1], [-beta/mu, -alpha/mu]])
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


probApproxDataInertia = []
pStarDataInertia = []
rateFunDataInertia = []
optNumberFailsDataInertia = []
detNadirDataInertia = []
JJFailsDataInertia = []
for mu in rangeInertiaValues:
    tempData = []
    tempData2 = []
    solutionType = []
    nadirTime = []
    
    deterministicNadir = 0
    numberFails_JJ = 0.0

    while deterministicNadir > -gamma:
        numberFails_JJ += 1.0
        b_deterministic = np.array([[0],[delta*numberFails_JJ/mu]])
        C1_deterministic = np.concatenate((A_deterministic,b_deterministic),axis=1)
        C2_deterministic = np.concatenate((np.zeros((1,2)),np.zeros((1,1))),axis=1)
        C_deterministic = np.concatenate((C1_deterministic, C2_deterministic), axis=0)
        
        deterministic_sample_path = []
        for time in tt:
            foo = -np.array([0,1,0]).dot(expm(C_deterministic*time))
            deterministic_sample_path.append(foo[2])
            
        deterministicNadir = min(deterministic_sample_path)
    detNadirDataInertia.append(deterministicNadir)
    JJFailsDataInertia.append(numberFails_JJ)
    
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
    
    
    for numberFailures in rangeNumberFailures:
        
        
        def objective(c):
            y0 = np.array([[0], [0], [-delta*numberFailures/mu],  [c[0]], [c[1]], [c[2]]])
            foo = (0.5*(np.transpose(y0).dot(BT.dot(y0)))/(sigma**2/epsilon))-numberFailures*epsilon*math.log(arrivalRate)            
            return foo.item(0)
        
        def constraint(c):
            y0 = np.array([[0], [0], [-delta*numberFailures/mu], [c[0]], [c[1]], [c[2]]])
            foo = expm(A*planningHorizon).dot(y0)
            return -foo.item(1)-gamma
    
        con = {'type': 'eq', 'fun': constraint}
        
        cInitial = np.array([[c1], [c2], [c3]])
              
        

        sol = minimize(objective, cInitial, constraints=con, method='SLSQP',options={'disp': False})
        
        tempData2.append(sol.x)
        
        optVal1 = sol.fun
        tempData.append(optVal1)

    optimalNumberFailures_JSNE = np.argmin(tempData)
    cOpt_JSNE = tempData2[optimalNumberFailures_JSNE]
    optVal_JSNE = tempData[optimalNumberFailures_JSNE]
    optSolType_JSNE = tempData[optimalNumberFailures_JSNE]
    
    optVal_JJ = -numberFails_JJ*epsilon*math.log(arrivalRate)
    
    optVals = [optVal_JJ, optVal_JSNE]
    optNumberFails = [numberFails_JJ, optimalNumberFailures_JSNE]
    
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

    probApproxDataInertia.append(math.exp(-min(optVals)/epsilon))
    pStarDataInertia.append(pStar)
    rateFunDataInertia.append(min(optVals))
    optNumberFailsDataInertia.append(optNumberFails)
    print(int(100*mu/inertiaMax), '%')
    
pStarDataInertia2 = np.array(pStarDataInertia)
pStarDataInertia3 = np.array(pStarDataInertia)
pStarDataInertia2 = np.insert(pStarDataInertia2,0,pStarDataInertia2[0])
pStarDataInertia3[abs(np.diff(pStarDataInertia2))>0.03] = np.nan

rateFunDataInertia2 = np.array(rateFunDataInertia)
rateFunDataInertia3 = np.array(rateFunDataInertia)
rateFunDataInertia2 = np.insert(rateFunDataInertia2,0,rateFunDataInertia2[0])
rateFunDataInertia3[abs(np.diff(rateFunDataInertia2))>0.2] = np.nan

JJFailsDataInertia2 = np.array(JJFailsDataInertia)
JJFailsDataInertia3 = np.array(JJFailsDataInertia)
JJFailsDataInertia2 = np.insert(JJFailsDataInertia2,0,JJFailsDataInertia2[0])
JJFailsDataInertia3[abs(np.diff(JJFailsDataInertia2))>0.5] = np.nan

optNumberFailsDataInertia2 = np.array(optNumberFailsDataInertia)
optNumberFailsDataInertia3 = np.array(optNumberFailsDataInertia)
optNumberFailsDataInertia2 = np.insert(optNumberFailsDataInertia2,0,optNumberFailsDataInertia2[0])
optNumberFailsDataInertia3[abs(np.diff(optNumberFailsDataInertia2))>0.5] = np.nan

detNadirDataInertia2 = np.array(detNadirDataInertia)
detNadirDataInertia3 = np.array(detNadirDataInertia)
detNadirDataInertia2 = np.insert(detNadirDataInertia2,0,optNumberFailsDataInertia2[0])
detNadirDataInertia3[abs(np.diff(detNadirDataInertia2))>0.03] = np.nan


fig, ax1 = plt.subplots()
ax1.set_xlabel('mu')
ax1.plot(range(len(rateFunDataInertia3)),rateFunDataInertia3,color="black")
ax1.set_ylabel('J')
ax2 = ax1.twinx()
ax2.set_xlabel('mu')
ax2.plot(range(len(detNadirDataInertia3)),detNadirDataInertia3,color="blue",linestyle='dotted')
ax2.set_ylabel('gamma-')

#tikzplotlib.save("Ex3_Inertia1v2.tex")

fig, ax3 = plt.subplots()

ax3.plot(range(len(pStarDataInertia3)),pStarDataInertia3,color="black")
ax3.set_ylabel('p*(T)', color="black")
ax4 = ax3.twinx()
ax4.plot(range(len(JJFailsDataInertia3)),JJFailsDataInertia3,color="blue", linestyle='dotted')
ax4.plot(range(len(optNumberFailsDataInertia3)),optNumberFailsDataInertia3,color="orange",linestyle='dashed')
ax4.set_ylabel('k*', color="green")
#tikzplotlib.save("Ex3_Inertia2v2.tex")

ax4.tick_params(axis='y', labelcolor="green")

