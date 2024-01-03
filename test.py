import numpy as np 
from financialmath.model.americanquadratic import (
    QuadraticApproximationAmericanVanilla)
from financialmath.model.blackscholes.closedform import ClosedFormBlackScholesInput
from financialmath.instruments.option import *
from financialmath.pricing.option.pde import (
    PDEBlackScholesValuation, 
    PDEBlackScholesParameters
)

S = 110
r = 0.12
q = 0
t = 0.25
sigma = 0.2
K = 100
put = False 
fut = True 
if put: option_type = OptionalityType.put
else: option_type = OptionalityType.call 
opt_payoff = OptionPayoff(
    option_type=option_type,
    exercise=ExerciseType.american,
    future=fut)
opt_spec = OptionSpecification(K, OptionTenor(expiry=t))
option = Option(opt_spec, opt_payoff)
inputdata = ClosedFormBlackScholesInput(S=S, r=r, q=q, sigma=sigma, t=t, K=K)
inputpde = PDEBlackScholesParameters(S=S, r=r, q=q, sigma=sigma, number_steps=200)
qame = QuadraticApproximationAmericanVanilla(inputdata=inputdata, put=put, future=fut)
pdepricer = PDEBlackScholesValuation(
    option=option, 
    parameters=inputpde, 
    first_order_greek=False, second_order_greek=False, 
    third_order_greek=False, max_workers=8)
pdetest = pdepricer.valuation()
print(pdetest.price)
print(qame.compute_prices())


#qame.initial_optimal_exercise_price()

#qame.S_star
