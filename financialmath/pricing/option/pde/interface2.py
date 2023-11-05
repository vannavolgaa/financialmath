from dataclasses import dataclass
import numpy as np
from typing import List
from financialmath.instruments.option import Option
from financialmath.pricing.option.schema import ImpliedOptionMarketData, OptionValuationResult
from financialmath.pricing.option.pde.framework.scheme import (OneFactorScheme, OneFactorSchemeList)
from financialmath.pricing.option.pde.framework.grid2 import (OptionRecursiveGrid, 
OptionPriceGrids, PDEPricing)
from financialmath.marketdata.schema import (OptionVolatilityPoint, 
VolatilityType, OptionVolatilitySurface, MoneynessType)
from financialmath.quanttool import QuantTool

@dataclass
class PDEBlackScholesPricerObject: 

    option : Option 
    marketdata : ImpliedOptionMarketData
    M : int = 100
    N : int = 400
    use_thread : bool = True
    numerical_scheme : OneFactorScheme = OneFactorSchemeList.implicit
    sensitivities : bool = True
    vol_bump_size: float = 0.01 
    spot_bump_size: float = 0.01 
    r_bump_size: float = 0.01 
    q_bump_size: float  = 0.01 

    method_name = "PDE solver of Black Scholes equation"

    def __post_init__(self): 
        self.S = self.marketdata.S
        self.sigma = self.marketdata.sigma
        self.t = self.option.specification.tenor.expiry
        self.r = self.marketdata.r
        self.q = self.marketdata.q
        self.scheme = OneFactorScheme.get_scheme(scheme=self.numerical_scheme)
        self.dt = self.time_step()
        self.dx = self.spot_logstep()
        
    @staticmethod
    def generate_spot_vector(dx: float, S: float, M : int) -> np.array: 
        spotvec = np.empty(M)
        spotvec[0] = S*np.exp((-dx*M/2))
        for i in range(1,M): 
            spotvec[i] = spotvec[i-1]*np.exp(dx)
        return spotvec

    def get_pde_method(self) -> str: 
        return self.method_name + ' w/ ' + self.numerical_scheme.value

    def get_volsurface_object(self, sigma:float) -> OptionVolatilitySurface: 
        list_spot = list(self.generate_spot_vector(self.dx,self.S,self.M))
        list_t = list(np.cumsum(np.repeat(self.dt, self.N)))
        volatility_points = []
        for tt in list_t: 
            for ss in list_spot: 
                point = OptionVolatilityPoint(
                    t=tt,moneyness=ss,sigma=sigma, 
                    volatility_type=VolatilityType.implied_volatility, 
                    moneyness_type=MoneynessType.strike)
                volatility_points.append(point)
        return OptionVolatilitySurface(points = volatility_points)
    
    def time_step(self) -> float: 
        return self.t/self.N

    def spot_logstep(self) -> float: 
        return self.sigma * np.sqrt(2*self.dt)

    def get_recursive_grid(self, sigma:float, r:float, q:float,
                           name : str) -> np.array:
        scheme = self.scheme
        volsurface = self.get_volsurface_object(sigma=sigma)
        dt = self.dt
        dx = self.dx
        N = self.N
        schemeobj = scheme(dx=dx, dt=dt, q=q, r=r,
                           volatility_surface=volsurface, N=N)
        matrixes = schemeobj.transition_matrixes()
        recursive_grid = OptionRecursiveGrid(
                        option=self.option, 
                        transition_matrixes = matrixes, 
                        S = self.marketdata.S, 
                        dx = self.dx, 
                        M = self.M)
        return {name: recursive_grid.generate_grid()}
    
    def generate_grid(self, arg_list:List[tuple], result:dict) -> dict: 
        if self.use_thread:
            data = QuantTool.send_tasks_with_threading(
                    self.get_recursive_grid, 
                    arg_list)
            [result.update(d[0]) for d in data]
        else: 
            data = []
            for a in arg_list: 
                data.append(self.get_recursive_grid(a[0], a[1], a[2], a[3])) 
            [result.update(d) for d in data]
        return result
            
    def generate_grid_object(self) -> OptionPriceGrids:
        sigma, r, q = self.sigma, self.r, self.q
        bump_vol, bump_r, bump_q = self.vol_bump_size, self.r_bump_size,\
                                    self.q_bump_size
        arg_list = [(sigma, r, q, 'no_bump',)]
        other_arg = [(sigma + bump_vol, r, q, 'vol_up_bump',), 
                     (sigma - bump_vol, r, q, 'vol_down_bump',), 
                     (sigma, r + bump_r, q, 'r_up_bump',), 
                     (sigma, r, q+bump_q, 'q_up_bump',)]
        if self.sensitivities: 
            arg_list = arg_list+other_arg
            result = {}
        else: 
            result = {'vol_up_bump' : None,'vol_down_bump':None, 
                    'r_up_bump':None, 'q_up_bump':None}
        result = self.generate_grid(arg_list,result)
        spot_vector = self.generate_spot_vector(dx=self.dx,S=self.S,M=self.M)
        return OptionPriceGrids(
            initial=result['no_bump'], vol_up=result['vol_up_bump'], 
            vol_down=result['vol_down_bump'], r_up=result['r_up_bump'], 
            q_up=result['q_up_bump'], spot_bump_size=self.spot_bump_size, 
            volatility_bump_size=bump_vol, r_bump_size=bump_r, 
            q_bump_size=bump_q, spot_vector=spot_vector)
    
    def valuation(self) -> OptionValuationResult: 
        valuation = PDEPricing(grid = self.generate_grid_object(), 
                               S=self.S)
        return OptionValuationResult(instrument=self.option, 
                                    marketdata=self.marketdata, 
                                    price=valuation.price(), 
                                    sensitivities=valuation.get_greeks(n=1), 
                                    method=self.get_pde_method())