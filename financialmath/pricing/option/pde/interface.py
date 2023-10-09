from dataclasses import dataclass
import numpy as np
from typing import List
from financialmath.instruments.option import Option
from financialmath.pricing.option.obj import ImpliedOptionMarketData, OptionValuationResult
from financialmath.pricing.option.pde.framework.scheme import (OneFactorScheme, OneFactorSchemeList)
from financialmath.pricing.option.pde.framework.grid import (RecursiveGrid, 
GridObject, BumpGrid, PricingGrid)

@dataclass
class PDEBlackScholesObject: 

    option : Option 
    marketdata : ImpliedOptionMarketData
    M : int = 100
    N : int = 400
    numerical_scheme : OneFactorSchemeList = OneFactorSchemeList.implicit
    bump : BumpGrid = BumpGrid()
    sensitivities : bool = True

    method_name = "PDE solver of Black Scholes equation"

    def __post_init__(self): 
        self.scheme = OneFactorScheme.get_scheme(scheme=self.numerical_scheme)
        self.dt = self.time_step()
        self.dx = self.spot_logstep()
        self.spotvec = self.spot_vector()
        self.manage_greeks()

    def manage_greeks(self): 
        if not self.sensitivities: 
            self.bump.volatility_up = False
            self.bump.volatility_down= False
            self.bump.r_up = False
            self.bump.q_up = False

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

    def spot_vector(self) -> np.array: 
        M = self.M
        spotvec = np.empty(M)
        S = self.marketdata.S
        spotvec[0] = S*np.exp((-self.dx*M/2))
        for i in range(1,M): 
            spotvec[i] = spotvec[i-1]*np.exp(self.dx)
        return spotvec

    def recursive_grid(self, sigma:float, r:float, q:float) -> np.array: 
        scheme = self.scheme
        volmatrix = self.volatility_matrix(sigma=sigma)
        dt = self.dt
        dx = self.dx
        S = self.marketdata.S
        N = self.N
        matrixes = scheme(dx=dx, dt=dt, q=q, r=r, S=S, 
                        volmatrix=volmatrix, N=N).transition_matrixes()
        return RecursiveGrid(self.option, matrixes, 
                            self.spotvec).generate()
    
    def generate_grids(self) -> GridObject: 
        sigma = self.marketdata.sigma 
        r = self.marketdata.r
        q = self.marketdata.q
        initial = self.recursive_grid(sigma=sigma, r=r, q=q)
        grids = GridObject(initial=initial, bump = self.bump) 
        if self.bump.volatility_up: 
            h = self.bump.volatility_bump_size
            grids.vol_up =  self.recursive_grid(sigma=sigma+h, r=r, q=q)
        if self.bump.volatility_down: 
            h = self.bump.volatility_bump_size
            grids.vol_down =  self.recursive_grid(sigma=sigma-h, r=r, q=q)
        if self.bump.r_up: 
            h = self.bump.r_bump_size
            grids.r_up =  self.recursive_grid(sigma=sigma, r=r+h, q=q)
        if self.bump.q_up: 
            h = self.bump.q_bump_size
            grids.q_up =  self.recursive_grid(sigma=sigma, r=r, q=q+h)
        return grids

    def pricing(self) -> OptionValuationResult: 
        valuation = PricingGrid(S = self.marketdata.S, spot_vector = self.spotvec, 
                                dt = self.dt, grid = self.generate_grids())
        
        return OptionValuationResult(instrument=self.option, 
                                    marketdata=self.marketdata, 
                                    price=valuation.price(), 
                                    sensitivities=valuation.greeks(), 
                                    method=self.get_method())


@dataclass
class PDEBlackScholes:

    option : Option or List[Option] 
    marketdata : ImpliedOptionMarketData or List[ImpliedOptionMarketData]
    M : int = 100
    N : int = 400
    numerical_scheme : OneFactorSchemeList = OneFactorSchemeList.implicit
    bump : BumpGrid = BumpGrid()
    sensitivities : bool = True

    def __post_init__(self): 
        if not isinstance(self.option, list):
            self.marketdata = [self.marketdata]
            self.option = [self.option]
    
    def main(self) -> OptionValuationResult or List[OptionValuationResult] : 
        output = [PDEBlackScholesObject(
            option=o, marketdata=m, M =self.M, 
            N = self.N, numerical_scheme=self.numerical_scheme,
            bump=self.bump, sensitivities=self.sensitivities
                                        ).pricing()
                for o,m in zip(self.option, self.marketdata)]
        if len(output == 1): return output[0]
        else: return output




