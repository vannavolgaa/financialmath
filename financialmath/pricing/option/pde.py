from dataclasses import dataclass, asdict
import time
import numpy as np
from typing import NamedTuple
from financialmath.instruments.option import Option 
from financialmath.tools.tool import MainTool
from financialmath.model.blackscholes.pde import (
    PDEBlackScholes,
    PDEBlackScholesInput
    )
from financialmath.pricing.numericalpricing.option.pde import (
    OneFactorOptionPriceGrids, 
    OneFactorOptionPriceGridGenerator
    )
from financialmath.pricing.schemas import (
    OptionGreeks, 
    OptionValuationResult
    )

class PDEBlackScholesParameters(NamedTuple):
    S : float 
    r : float 
    q : float 
    sigma : float 
    number_steps : int = 400
    spot_vector_size : int = 100
    dS : float = 0.01 
    dsigma : float = 0.01 
    dr : float = 0.01 
    dq : float = 0.01 

@dataclass
class PDEBlackScholesValuation: 
    option : Option 
    parameters : PDEBlackScholesParameters
    first_order_greek : bool = True
    second_order_greek : bool = True
    third_order_greek : bool = True
    max_workers : int = 4

    def __post_init__(self): 
        self.start = time.time()
        self.inputdata = PDEBlackScholesInput(
            S = self.parameters.S, 
            sigma = self.parameters.sigma, 
            r = self.parameters.r, 
            q = self.parameters.q, 
            number_steps = self.parameters.number_steps, 
            spot_vector_size = self.parameters.spot_vector_size, 
            dS = self.parameters.dS, 
            dsigma = self.parameters.dsigma, 
            dr = self.parameters.dr,
            dq = self.parameters.dq,
            t = self.option.specification.tenor.expiry,
            future = self.option.payoff.future, 
            max_workers=self.max_workers
        )
        self.pdebs = PDEBlackScholes(inputdata=self.inputdata).get(
            first_order_greek=self.first_order_greek, 
            second_order_greek=self.second_order_greek, 
            third_order_greek=self.third_order_greek
        )
        self.spot_vector = self.pdebs.spot_vector
        self.matrixes = self.pdebs.matrixes
        self.pricer = self.get_option_price_grids()

    def method(self): 
        return 'PDE solver of Black Scholes equation'
    
    def compute_option_price_grid(self, args:tuple) -> dict[int,np.array]:
        M, N = self.inputdata.spot_vector_size, self.inputdata.number_steps
        name, matrix = args[0], args[1]
        spot_vector = self.pdebs.spot_vector
        try: 
            fdm = OneFactorOptionPriceGridGenerator(
                option = self.option, 
                matrixes = matrix, 
                spot_vector=spot_vector, 
                ) 
            return {name:fdm.option_price_grid()}
        except Exception as e:
            return {name:np.reshape(np.repeat(np.nan, M*N),(M,N))}
    
    def get_option_price_grids(self) -> OneFactorOptionPriceGrids: 
        matrixes = asdict(self.matrixes)
        args_list = [[k, matrixes[k]] for k in list(matrixes.keys())]
        opgrids = MainTool.send_task_with_futures(
                task = self.compute_option_price_grid,
                args = args_list, 
                max_workers = self.max_workers
                )
        opgrids = MainTool.listdict_to_dictlist(opgrids)
        print()
        return OneFactorOptionPriceGrids(**opgrids)
    
    def price(self) -> float: 
        return self.pricer.price(
            S = self.parameters.S, 
            spot_vector = self.spot_vector)
    
    def greeks(self) -> OptionGreeks: 
        return OptionGreeks(**self.pricer.greeks(
            S = self.parameters.S, 
            spot_vector = self.spot_vector, 
            dS = self.parameters.dS,
            dr = self.parameters.dr,
            dq = self.parameters.dq,
            dsigma = self.parameters.dsigma,
            dt = self.pdebs.dt
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
        


    
    

    
    
    
        
