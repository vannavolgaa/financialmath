from financialmath.model.blackscholes.montecarlo import (BlackScholesDiscretization, 
MonteCarloBlackScholes, MonteCarloBlackScholesInput)
from financialmath.model.blackscholes.pde import (PDEBlackScholes, PDEBlackScholesInput)
import numpy as np 
import matplotlib.pyplot as plt 
from financialmath.instruments.option import *
from dataclasses import dataclass
from financialmath.pricing.option.pde import PDEBlackScholesValuation
import matplotlib.pyplot as plt
from financialmath.pricing.numericalpricing.option import MonteCarloPricing, MonteCarloLeastSquare, MonteCarloLeastSquareMethod
import time
from enum import Enum

lookback_payoff_floatK = LookBackPayoff(floating_strike=True, floating_spot=False, 
                                 spot_method=None, strike_method=LookbackMethod.geometric_mean, 
                                 spot_observation=None, strike_observation=ObservationType.continuous)
lookback_payoff_floatS = LookBackPayoff(floating_strike=False, floating_spot=True, 
                                 spot_method=LookbackMethod.geometric_mean, strike_method=None, 
                                 spot_observation=ObservationType.continuous, strike_observation=None)
bothfloat_lookback = LookBackPayoff(floating_strike=True, floating_spot=True, 
                                 spot_method=LookbackMethod.geometric_mean, strike_method=LookbackMethod.arithmetic_mean, 
                                 spot_observation=ObservationType.continuous, strike_observation=ObservationType.continuous)
opt_payoff = OptionPayoff(
    option_type=OptionalityType.call,
    exercise=ExerciseType.european, forward_start=True, lookback=lookback_payoff_floatS)
opt_spec = OptionSpecification(100, OptionTenor(expiry=1, bermudan=[0.5], forward_start=0.4), )
option = Option(opt_spec, opt_payoff)
S = 100 
r = 0.01
q = 0.1 
t = 1
sigma = 0.2
N = 100
M = 50000
dt = t/N
Bu = 120 

mcinput = MonteCarloBlackScholesInput(
    S=S, r=r,q=q,t=t,sigma=sigma,
    number_paths=M, 
    number_steps=N,
    discretization=BlackScholesDiscretization.milstein,
    max_workers=1)
bsmc = MonteCarloBlackScholes(inputdata=mcinput)
simulator = bsmc.get(False,False,False)
sim = simulator.sim
test = MonteCarloPricing(sim=sim, option=option, r=r)

test.compute_price()


    

