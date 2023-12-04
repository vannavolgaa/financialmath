from dataclasses import dataclass
import time
import numpy as np
from financialmath.instruments.option import Option 
from financialmath.model.blackscholes.pde import (PDEBlackScholes,
PDEBlackScholesInput, PDEBlackScholesOutput)
from financialmath.pricing.numericalpricing.option import (SpotFactorFiniteDifferencePricing, 
OptionPriceGrids, FiniteDifferencePricer)
from financialmath.pricing.schemas import OptionGreeks, OptionValuationResult
from financialmath.tools.tool import MainTool

@dataclass
class PDEBlackScholesValuation: 

    option : Option 
    inputdata : PDEBlackScholesInput
    use_futures_thread : bool = True

    def __post_init__(self): 
        self.start = time.time()
        self.output = PDEBlackScholes(inputdata=self.inputdata).get()
        self.pricer = FiniteDifferencePricer(self.option_price_grids())
        self.end = time.time()
    
    def method(self): 
        return 'PDE solver of Black Scholes equation'
    
    def compute_option_price_grid(self, args:tuple) -> dict[int,np.array]:
        M, N = self.inputdata.spot_vector_size, self.inputdata.number_steps
        grids, i = args[0], args[1]
        S, dS, dt = self.inputdata.S, self.output.dS, self.output.dt
        spot_vector = self.output.spot_vector
        try: 
            fdm = SpotFactorFiniteDifferencePricing(
                option = self.option, grid_list = grids, 
                spot_vector=spot_vector, S=S, dt=dt, dS=dS) 
            return {i:fdm.option_price_grid()}
        except: return {i:np.reshape(np.repeat(np.nan, M*N),(M,N))}
    
    def option_price_grids(self) -> OptionPriceGrids: 
        args_list = [(self.output.grid_list, 0),
                     (self.output.grid_list_sigma_up, 1), 
                     (self.output.grid_list_sigma_down, 2),
                     (self.output.grid_list_r_up, 3), 
                     (self.output.grid_list_q_up, 4),
                     (self.output.grid_list_sigma_uu, 5),
                     (self.output.grid_list_sigma_dd, 6)]
        if self.use_futures_thread: 
            opgrids = MainTool.send_task_with_futures(
                self.compute_option_price_grid,args_list)
        else: 
            opgrids = [self.compute_option_price_grid(a) for a in args_list]
        opgrids = MainTool.listdict_to_dictlist(opgrids)
        data = self.inputdata
        return OptionPriceGrids(
            initial=opgrids[0], 
            spot_vector = self.output.spot_vector, 
            S=data.S, dS = data.dS, dsigma=data.dsigma, 
            dt=self.output.dt, dr =data.dr, dq=data.dq,
            vol_up = opgrids[1],vol_down=opgrids[2], 
            vol_uu = opgrids[5], vol_dd = opgrids[6], 
            r_up = opgrids[3], q_up=opgrids[4])
    
    def price(self) -> float: 
        return self.pricer.price()
    
    def greeks(self) -> OptionGreeks: 
        return self.pricer.greeks()
    
    def valuation(self) -> OptionValuationResult: 
        return OptionValuationResult(
            instrument = self.option, method = self.method(), 
            price = self.price(), sensitivities = self.greeks(), 
            inputdata = self.inputdata, time_taken=self.end-self.start)


    
    
    
    

    
    
    
        
