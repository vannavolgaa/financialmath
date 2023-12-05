from financialmath.model.blackscholes.montecarlo import (BlackScholesDiscretization, 
MonteCarloBlackScholes, MonteCarloBlackScholesInput)
from financialmath.model.blackscholes.pde import (PDEBlackScholes, PDEBlackScholesInput)
import numpy as np 
import matplotlib.pyplot as plt 
from financialmath.instruments.option import *
from dataclasses import dataclass
from financialmath.pricing.option2.pde import PDEBlackScholesValuation


opt_payoff = OptionPayoff(
    option_type=OptionalityType.call,
    exercise=ExerciseType.european, 
    barrier_type=BarrierType.down_and_out, 
    barrier_observation=ObservationType.continuous, forward_start=True)
opt_spec = OptionSpecification(100, OptionTenor(expiry=1, forward_start=0.5), barrier_up=120, barrier_down=80, rebate=0)
option = Option(opt_spec, opt_payoff)


S = 100 
r = 0.01
q = 0.1 
t = 1 
sigma = 0.2
N = 400
M = 10000
dt = t/N
Bu = 120 

option_steps = option.specification.get_steps(N=N)

option_steps.forward_start

mcinput = MonteCarloBlackScholesInput(
    S=S, r=r,q=q,t=t,sigma=sigma,
    number_paths=M, 
    number_steps=N,
    discretization=BlackScholesDiscretization.milstein,
    max_workers=1)

bsmc = MonteCarloBlackScholes(inputdata=mcinput)
test = bsmc.get(False,False,False)
sim = test.sim

