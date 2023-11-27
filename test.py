from financialmath.model.blackscholes.montecarlo import (BlackScholesDiscretization, 
MonteCarloBlackScholes, MonteCarloBlackScholesInput)
from financialmath.model.blackscholes.pde import (PDEBlackScholes, PDEBlackScholesInput)
import numpy as np 
import matplotlib.pyplot as plt 

S = 100 
r = 0.01
q = 0.02 
t = 1 
sigma = 0.2
N = 500
M = 250
dt = t/N

mcinput = MonteCarloBlackScholesInput(
    S=S, r=r,q=q,t=t,sigma=sigma,
    number_paths=M, 
    number_steps=N,
    discretization=BlackScholesDiscretization.milstein,
    first_order_greek=True, 
    second_order_greek=False, 
    third_order_greek=False
)

pdeinput = PDEBlackScholesInput(
    S=S, r=r,q=q,t=t,sigma=sigma,
    spot_vector_size=M, 
    number_steps=N,
)

test = PDEBlackScholes(pdeinput).get()






