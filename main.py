from financialmath.instruments.option import *
from financialmath.pricing.option.pde import (
    PDEBlackScholesValuation, 
    PDEBlackScholesParameters
)
from financialmath.pricing.option.montecarlo import (
    MonteCarloBlackScholesValuation, 
    MonteCarloBlackScholesParameters, 
    BlackScholesDiscretization, 
    RandomGeneratorType
)
S = 100 
r = 0.01
q = 0.1
sigma = 0.5
opt_payoff = OptionPayoff(
    option_type=OptionalityType.call,
    exercise=ExerciseType.american,
    barrier_observation=ObservationType.continuous, 
    barrier_type=BarrierType.down_and_in)
opt_spec = OptionSpecification(100, OptionTenor(expiry=1, bermudan=[0.25,0.5,0.75]), barrier_up=105, barrier_down=95)
option = Option(opt_spec, opt_payoff)

inputpde = PDEBlackScholesParameters(
    S=S, 
    r=r, 
    q=q, 
    sigma=sigma)
inputmc = MonteCarloBlackScholesParameters(
    S=S, 
    r=r, 
    q=q, 
    sigma=sigma,
    number_paths=20000, 
    number_steps=100, 
    randoms_generator=RandomGeneratorType.quasiMC_halton, 
    dS = 1, 
    dsigma=0.1, 
    dr = 0.01, 
    dq = 0.01)

pdepricer = PDEBlackScholesValuation(
    option=option, 
    parameters=inputpde, 
    first_order_greek=True, second_order_greek=True, 
    third_order_greek=False, max_workers=8)
mcpricer = MonteCarloBlackScholesValuation(
    option=option, 
    parameters=inputmc, 
    first_order_greek=True, second_order_greek=True, 
    third_order_greek=False, max_workers=8)

pdetest = pdepricer.valuation()
mctest = mcpricer.valuation()
print(pdetest.price)
print(mctest.price)
print(pdetest.sensitivities)
print(mctest.sensitivities)



bool(0)