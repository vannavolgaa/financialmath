from financialmath.model.blackscholes.closedform import (
    BlackScholesEuropeanVanillaCall, 
    ClosedFormBlackScholesInput, 
    BlackScholesEuropeanVanillaPut, 
    EuropeanVanillaImpliedData, 
    EuropeanVanillaImpliedVolatility, 
    BlackEuropeanVanillaCall, 
    BlackEuropeanVanillaPut
    )
import numpy as np




S = np.array([100, 110, 95, 84]) 
K = np.array([80, 110, 120, 30]) 
r = np.array([0.01, 0.02, 0.03, 0.1]) 
t = np.array([1, 10, 0.03, 0.2]) 
q = np.array([0.005, 0.035, 0.15, 0.02]) 
sigma = np.array([0.2, 0.75, 0.25, 0.12]) 
F = S*np.exp((r-q)*t)

inputdata_fut = ClosedFormBlackScholesInput(
    S=S,K=K, r=r, q=q, sigma=sigma, t=t)
inputdata = ClosedFormBlackScholesInput(
    S=S,K=K, r=r, q=q, sigma=sigma, t=t)
call_fut = BlackEuropeanVanillaCall(inputdata_fut)
call = BlackScholesEuropeanVanillaCall(inputdata)
cprice = call.price()
cprice_fut = call_fut.price()
put = BlackScholesEuropeanVanillaPut(inputdata)
pprice = put.price()
put_fut = BlackEuropeanVanillaPut(inputdata_fut)
pprice_fut = put_fut.price()

test = EuropeanVanillaImpliedVolatility(
    price = cprice, 
    S = S, 
    K = K, 
    t=t, r=r, q=q, 
    call=True, future=False
).get()

test = EuropeanVanillaImpliedData(
    P = pprice, 
    C = cprice, 
    S = S, 
    K = K, 
    t=t, r=None, 
    carry_cost=False, future=False
).get()

test