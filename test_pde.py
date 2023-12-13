
from financialmath.model.blackscholes.montecarlo import (BlackScholesDiscretization, 
MonteCarloBlackScholes, MonteCarloBlackScholesInput)
from financialmath.model.blackscholes.pde import (PDEBlackScholes, PDEBlackScholesInput)
import numpy as np 
import matplotlib.pyplot as plt 
from financialmath.instruments.option import *
from dataclasses import dataclass
from financialmath.pricing.option.pde import PDEBlackScholesValuation
import matplotlib.pyplot as plt
from financialmath.pricing.numericalpricing.option import MonteCarloPricing


opt_payoff = OptionPayoff(
    option_type=OptionalityType.call,
    exercise=ExerciseType.european, 
    barrier_observation=ObservationType.in_fine, 
    barrier_type=BarrierType.up_and_out)

opt_spec = OptionSpecification(100, OptionTenor(expiry=1), barrier_up=120, barrier_down=80, rebate=0)
option = Option(opt_spec, opt_payoff)

inputpde = PDEBlackScholesInput(S=100, r=0.01, q=0.1, sigma=0.2, t=1)

pricer = PDEBlackScholesValuation(option=option, inputdata=inputpde)

test = pricer.valuation()

test.price

