from enum import Enum 
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np 
from typing import List
from scipy import interpolate
from financialmath.marketdata.schemas2 import VolatilitySurface
from financialmath.tools.tool import MainTool
from financialmath.model.svi import (
    StochasticVolatilityInspired, 
    SSVIFunctions
    )


class StrikeType(Enum): 
    strike = 1 
    moneyness = 2 
    log_moneyness = 3
    forward_log_moneyness = 4
    forward_moneyness = 5

@dataclass
class ExtrapolatedTotalVarianceTermStructure: 
    t : List[float]
    totvariance : List[float]
    max_t : int = 100

    interpolation_method = 'cubic'

    def __post_init__(self): 
        ordered_dict = MainTool.order_join_lists(
            keys=self.t, 
            values=self.totvariance)
        self.t = np.array(list(ordered_dict.keys()))
        self.total_variance = np.array(list(ordered_dict.values()))
        self.extrapolator = self.extrapolation()

    def extrapolation(self) -> interpolate.interp1d: 
        tvar, t = self.totvariance, self.t
        n = len(self.tvar)
        tvar2, t2, tvar1, t1 = tvar[n-1], t[n-1], tvar[n-2], t[n-2]
        slope = (tvar2-tvar1)/(t2-t1)
        max_tvar = slope*self.max_t
        tvar = np.insert(tvar, [0, len(tvar)], [0, max_tvar])
        t = np.insert(t, [0, len(t)], [0, self.max_t])
        return interpolate.interp1d(t, tvar, kind = self.interpolation_method)
    
    def total_variance(self, t: np.array) -> np.array: 
        return self.extrapolator(x=t)


class FlatVolatilitySurface(VolatilitySurface): 

    def __init__(self, volatility: float): 
        self.volatility = volatility 

    def implied_variance(self, k: np.array, t: np.array) -> np.array:
        n = len(k)*len(t)
        return np.repeat(self.volatility**2,n)
    
    def total_variance(self, k: np.array, t: np.array) -> np.array:
        n = len(k)
        return np.repeat(t*(self.volatility**2),n)
    
    def implied_volatility(self, k: np.array, t: np.array) -> np.array:
        n = len(k)*len(t)
        return np.repeat(self.volatility,n)
    
    def local_volatility(self, k: np.array, t: np.array) -> np.array:
        n = len(k)*len(t)
        return np.repeat(self.volatility,n)
    
    def skew(self, k: np.array, t: np.array) -> np.array:
        n = len(k)*len(t)
        return np.repeat(0,n)

@dataclass
class PowerLawSSVIVolatilitySurface(VolatilitySurface): 

    nu : float 
    rho : float 
    _gamma : float 
    t : List[float]
    totvariance : List[float]
    ssvi_function: SSVIFunctions = SSVIFunctions.power_law1

    def __post_init__(self): 
        self.atm_volatility_term_structure = \
            ExtrapolatedTotalVarianceTermStructure(
                t = self.t,
                totvariance = self.totvariance
            )

    def strike_type(self) -> StrikeType: 
        return StrikeType.forward_log_moneyness
    
    def svi(self, t: float) -> StochasticVolatilityInspired: 
        atm_tvar = self.atm_volatility_term_structure.total_variance(t=t)
    
    def power_law(self, atmtvar: np.array) -> np.array: 
        return self.ssvi_function(atmtvar=atmtvar, _nu=self.nu, 
                                  _gamma = self._gamma)

    def total_variance(self, k: np.array, t: np.array) -> np.array:
        atm_tvar = self.atm_volatility_term_structure.total_variance(t=t)
        pl = self.power_law(atm_tvar)
        term1 = self.p*pl*k
        term2 = np.sqrt((pl*k+self.p)**2 + (1-self.p**2))
        return .5*atm_tvar*(1 + term1 + term2)
    
    def implied_variance(self, k: np.array, t: np.array) -> np.array:
        return self.total_variance(k=k, t=t)/t 
    
    def implied_volatility(self, k: np.array, t: np.array) -> np.array:
        return np.sqrt(self.implied_variance(k=k, t=t))

    

