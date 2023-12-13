from dataclasses import dataclass
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
    OneFactorFiniteDifferenceGreeks,
    OneFactorFiniteDifferencePricer,
    OneFactorOptionPriceGrids
    )
from financialmath.pricing.schemas import (
    OptionGreeks, 
    OptionValuationResult
    )

class PDEBlackScholesParameters(NamedTuple):
    pass

@dataclass
class PDEBlackScholesValuation: 

    option : Option 
    inputdata : PDEBlackScholesInput
    first_order_greek : bool = True
    second_order_greek : bool = True
    third_order_greek : bool = True
    max_workers : int = 4

    def __post_init__(self): 
        self.start = time.time()
        self.pdebs = PDEBlackScholes(inputdata=self.inputdata).get(
            first_order_greek=self.first_order_greek, 
            second_order_greek=self.second_order_greek,
            third_order_greek=self.third_order_greek
        )
        self.op_grids = self.option_price_grids()
        self.pricer = OneFactorFiniteDifferenceGreeks(self.op_grids)
    
    def method(self): 
        return 'PDE solver of Black Scholes equation'
    
    def compute_option_price_grid(self, args:tuple) -> dict[int,np.array]:
        M, N = self.inputdata.spot_vector_size, self.inputdata.number_steps
        grids, i = args[0], args[1]
        S, dS, dt = self.inputdata.S, self.pdebs.dS, self.pdebs.dt
        spot_vector = self.pdebs.spot_vector
        try: 
            fdm = OneFactorFiniteDifferencePricer(
                option = self.option, 
                grid_list = grids, 
                spot_vector=spot_vector, 
                S=S, 
                dt=dt, 
                dS=dS
                ) 
            return {i:fdm.option_price_grid()}
        except: return {i:np.reshape(np.repeat(np.nan, M*N),(M,N))}
    
    def option_price_grids(self) -> OneFactorOptionPriceGrids: 
        args_list = [(self.pdebs.grid_list, 0),
                     (self.pdebs.grid_list_sigma_up, 1), 
                     (self.pdebs.grid_list_sigma_down, 2),
                     (self.pdebs.grid_list_r_up, 3), 
                     (self.pdebs.grid_list_q_up, 4),
                     (self.pdebs.grid_list_sigma_uu, 5),
                     (self.pdebs.grid_list_sigma_dd, 6)]
        opgrids = MainTool.send_task_with_futures(
                task = self.compute_option_price_grid,
                args = args_list, 
                max_workers = self.max_workers
                )
        opgrids = MainTool.listdict_to_dictlist(opgrids)
        return OneFactorOptionPriceGrids(
            initial = opgrids[0], 
            spot_vector = self.pdebs.spot_vector, 
            S = self.inputdata.S, 
            dS = self.inputdata.dS, 
            dsigma = self.inputdata.dsigma, 
            dt = self.pdebs.dt, 
            dr = self.inputdata.dr, 
            dq = self.inputdata.dq,
            vol_up = opgrids[1],
            vol_down=opgrids[2], 
            vol_uu = opgrids[5], 
            vol_dd = opgrids[6], 
            r_up = opgrids[3], 
            q_up = opgrids[4]
            )
    
    def price(self) -> float: 
        return self.pricer.price()
    
    def greeks(self) -> OptionGreeks: 
        g = self.pricer.greeks()
        return OptionGreeks(**g)
    
    def valuation(self) -> OptionValuationResult: 
        return OptionValuationResult(
            instrument = self.option, 
            method = self.method(), 
            price = self.price(), 
            sensitivities = self.greeks(), 
            inputdata = self.inputdata, 
            time_taken = time.time()-self.start
            )


    
    
    
    

    
    
    
        
