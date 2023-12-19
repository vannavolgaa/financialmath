import numpy as np 
from financialmath.model.americanquadratic import (
    QuadraticApproximationAmericanVanillaCall)
from financialmath.model.blackscholes.closedform import ClosedFormBlackScholesInput

S = 100 
r = -0.01
q = 0.16
t = 1
sigma = np.array([0.2,0.5,0.49,0.48,0.21,0.27,0.39,0.255])
K = 100 
inputdata = ClosedFormBlackScholesInput(S=S, r=r, q=q, sigma=sigma, t=t, K=K)
qame = QuadraticApproximationAmericanVanillaCall(inputdata=inputdata)
test = qame.compute_prices()


