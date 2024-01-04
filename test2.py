from financialmath.model.blackscholes.closedform import BlackScholesEuropeanVanillaCall, ClosedFormBlackScholesInput, BlackScholesEuropeanVanillaPut
import numpy as np 

S = 100 
K = 150
r = 0.01 
t = 1 
q = 0.05 
sigma = 0.758

inputdata = ClosedFormBlackScholesInput(
    S=S,K=K, r=r, q=q, sigma=sigma, t=t)
call = BlackScholesEuropeanVanillaCall(inputdata)
cprice = call.price()
put = BlackScholesEuropeanVanillaPut(inputdata)
pprice = put.price()

dK = K*np.exp(-r*t)
term1 = np.sqrt(2*np.pi)/(S+dK)
term2 = (S-dK)/2
term3 = ((S-dK)**2)/np.pi 
term4 = cprice - term2 
sigma_0 = (term1 + term4 + np.sqrt(np.maximum(term4**2 - term3,0)))/(100*np.sqrt(t))


