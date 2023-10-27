from dataclasses import dataclass
import numpy as np
from typing import List
from financialmath.instruments.option import Option
from financialmath.pricing.option.schema import ImpliedOptionMarketData, OptionValuationResult
from financialmath.pricing.option.pde.framework.scheme import (OneFactorScheme, OneFactorSchemeList)
from financialmath.pricing.option.pde.framework.grid import (OptionRecursiveGrid, 
OptionPriceGrids, PricingGrid)

@dataclass
class PDEBlackScholesObject: 

    option : Option 
    marketdata : ImpliedOptionMarketData
    M : int = 100
    N : int = 400
    numerical_scheme : OneFactorSchemeList = OneFactorSchemeList.implicit
    sensitivities : bool = True
    vol_bump_size: float = 0.01 
    spot_bump_size: float = 0.01 
    r_bump_size: float = 0.01 
    q_bump_size: float  = 0.01 

    method_name = "PDE solver of Black Scholes equation"

    def __post_init__(self): 
        self.scheme = OneFactorScheme.get_scheme(scheme=self.numerical_scheme)
        self.dt = self.time_step()
        self.dx = self.spot_logstep()

    def get_method(self) -> str: 
        return self.method_name + ' w/ ' + self.numerical_scheme.value

    def volatility_matrix(self, sigma:float) -> np.array: 
        M = self.M 
        N = self.N 
        return np.reshape(np.repeat(sigma, M*N), (M,N))
    
    def time_step(self) -> float: 
        t = self.option.specification.tenor.expiry
        return t/self.N

    def spot_logstep(self) -> float: 
        sigma = self.marketdata.sigma 
        return sigma * np.sqrt(2*self.dt)

    def generate_recursive_grid(self, sigma:float, r:float, q:float) -> np.array: 
        scheme = self.scheme
        volmatrix = self.volatility_matrix(sigma=sigma)
        dt = self.dt
        dx = self.dx
        N = self.N
        matrixes = scheme(dx=dx, dt=dt, q=q, r=r, 
                        volmatrix=volmatrix, N=N).transition_matrixes()
        recursive_grid = OptionRecursiveGrid(
                        option=self.option, 
                        transition_matrixes = matrixes, 
                        S = self.marketdata.S, 
                        dx = self.dx, 
                        M = self.M)
        return recursive_grid.generate_classic_option_grid()

    def generate_grids(self) -> OptionPriceGrids: 
        sigma = self.marketdata.sigma 
        r = self.marketdata.r
        q = self.marketdata.q
        initial = self.generate_recursive_grid(sigma=sigma, r=r, q=q)
        grids = OptionPriceGrids(initial=initial, volatility_bump_size=self.vol_bump_size, 
                                q_bump_size=self.q_bump_size, r_bump_size= self.r_bump_size, 
                                spot_bump_size=self.spot_bump_size) 
        if self.sensitivities: 
            vol_h = self.vol_bump_size
            vol_r = self.r_bump_size
            vol_q = self.q_bump_size
            grids.vol_up =  self.generate_recursive_grid(sigma=sigma+vol_h, r=r, q=q)
            grids.vol_down =  self.generate_recursive_grid(sigma=sigma-vol_h, r=r, q=q)
            grids.r_up =  self.generate_recursive_grid(sigma=sigma, r=r+vol_r, q=q)
            grids.q_up =  self.generate_recursive_grid(sigma=sigma, r=r, q=q+vol_q)
        return grids

    def pricing(self) -> OptionValuationResult: 
        valuation = PricingGrid(S = self.marketdata.S, dx = self.dx, M=self.M,
                                dt = self.dt, grid = self.generate_grids())
        
        return OptionValuationResult(instrument=self.option, 
                                    marketdata=self.marketdata, 
                                    price=valuation.price(), 
                                    sensitivities=valuation.greeks(), 
                                    method=self.get_method())

@dataclass
class PDEBlackScholes:

    option: Option or List[Option] 
    marketdata: ImpliedOptionMarketData or List[ImpliedOptionMarketData]
    M: int = 100
    N: int = 400
    numerical_scheme: OneFactorSchemeList = OneFactorSchemeList.implicit
    sensitivities: bool = True
    vol_bump_size: float = 0.01 
    spot_bump_size: float = 0.01 
    r_bump_size: float = 0.01 
    q_bump_size: float  = 0.01 

    def __post_init__(self): 
        if not isinstance(self.option, list):
            self.marketdata = [self.marketdata]
            self.option = [self.option]
    
    def main(self) -> OptionValuationResult or List[OptionValuationResult] : 
        output = [PDEBlackScholesObject(
            option=o, marketdata=m, M =self.M, 
            N = self.N, numerical_scheme=self.numerical_scheme,
            sensitivities=self.sensitivities).pricing()
                for o,m in zip(self.option, self.marketdata)]
        if len(output)==1: return output[0]
        else: return output




