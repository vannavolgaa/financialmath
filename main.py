from financialmath.instruments.option import *
from financialmath.pricing.option.pde import (
    PDEBlackScholesValuation, 
    PDEBlackScholesParameters
)
from financialmath.pricing.option.montecarlo import (
    MonteCarloBlackScholesValuation, 
    MonteCarloBlackScholesParameters
)

opt_payoff = OptionPayoff(
    option_type=OptionalityType.call,
    exercise=ExerciseType.european)
opt_spec = OptionSpecification(100, OptionTenor(expiry=1), barrier_up=120, barrier_down=80, rebate=0)
option = Option(opt_spec, opt_payoff)

inputpde = PDEBlackScholesParameters(S=100, r=0.01, q=0.1, sigma=0.2)
inputmc = MonteCarloBlackScholesParameters(S=100, r=0.01, q=0.1, sigma=0.2,
                                           number_paths=10000)

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

pdetest.sensitivities
mctest.sensitivities