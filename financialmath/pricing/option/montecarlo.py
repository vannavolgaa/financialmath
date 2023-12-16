from dataclasses import dataclass, asdict
import time
import numpy as np
from typing import NamedTuple
from financialmath.instruments.option import Option 
from financialmath.tools.tool import MainTool
from financialmath.model.blackscholes.montecarlo import (
    MonteCarloBlackScholes,
    MonteCarloBlackScholesInput, 
    BlackScholesDiscretization
    )
from financialmath.pricing.numericalpricing.option.montecarlo import (
    MonteCarloPricing,
    MonteCarloPrice
    )
from financialmath.pricing.schemas import (
    OptionGreeks, 
    OptionValuationResult
    )
from financialmath.tools.simulation import ( 
    RandomGeneratorType)

class MonteCarloBlackScholesParameters(NamedTuple):
    S : float 
    r : float 
    q : float 
    sigma : float 
    number_steps : int = 400
    number_paths : int = 10000
    randoms_generator : RandomGeneratorType = \
        RandomGeneratorType.antithetic
    discretization: BlackScholesDiscretization = \
        BlackScholesDiscretization.euler
    dS : float = 0.01 
    dsigma : float = 0.01 
    dr : float = 0.01 
    dq : float = 0.01 
    max_workers : int = 4

@dataclass
class MonteCarloBlackScholesValuation: 

    option : Option 
    parameters : MonteCarloBlackScholesParameters
    first_order_greek : bool = True
    second_order_greek : bool = True
    third_order_greek : bool = True
    max_workers : int = 4

    def __post_init__(self): 
        self.start = time.time()
        self.inputdata = MonteCarloBlackScholesInput(
            S = self.parameters.S, 
            sigma = self.parameters.sigma, 
            r = self.parameters.r, 
            q = self.parameters.q, 
            randoms_generator = self.parameters.randoms_generator,
            discretization = self.parameters.discretization, 
            number_paths = self.parameters.number_paths, 
            number_steps = self.parameters.number_steps,
            dS = self.parameters.dS, 
            dsigma = self.parameters.dsigma, 
            dr = self.parameters.dr,
            dq = self.parameters.dq,
            t = self.option.specification.tenor.expiry,
            future = self.option.payoff.future, 
            max_workers=self.max_workers
        )
        self.mcbs = MonteCarloBlackScholes(inputdata=self.inputdata).get(
            first_order_greek = self.first_order_greek, 
            second_order_greek = self.second_order_greek, 
            third_order_greek = self.third_order_greek   
        )
        self.simulations = self.mcbs.simulations
        self.pricer = self.get_monte_carlo_prices()

    def method(self): 
        return 'Black scholes SDE solver using Monte-Carlo'
    
    def compute_monte_carlo_price(self, args: tuple[str, np.array])\
          -> dict[str, float]: 
        name, sim = args[0], args[1]
        try: 
            mcp = MonteCarloPrice(
                sim = sim, 
                option = self.option, 
                r = self.parameters.r, 
                volatility_matrix = None
            ) 
            return {name : mcp.compute_price()} 
        except Exception as e: return {name : np.nan} 

    def get_monte_carlo_prices(self) -> MonteCarloPricing: 
        simulations = asdict(self.simulations)
        args_list = [[k, simulations[k]] for k in list(simulations.keys())]
        mcprices = MainTool.send_task_with_futures(
                task = self.compute_monte_carlo_price,
                args = args_list, 
                max_workers = self.max_workers
                )
        mcprices = MainTool.listdict_to_dictlist(mcprices)
        return MonteCarloPricing(**mcprices)
    
    def price(self) ->float: 
        return self.pricer.price()
    
    def greeks(self) ->OptionGreeks: 
        return OptionGreeks(**self.pricer.greeks(
            dS = self.parameters.dS,
            dr = self.parameters.dr,
            dq = self.parameters.dq,
            dsigma = self.parameters.dsigma,
            dt = self.mcbs.dt
        ))
    
    def valuation(self) -> OptionValuationResult: 
        return OptionValuationResult(
            instrument = self.option, 
            method = self.method(), 
            price = self.price(), 
            sensitivities = self.greeks(), 
            inputdata = self.parameters,
            time_taken = time.time()-self.start
            )