from dataclasses import dataclass
import numpy as np 
from enum import Enum
from typing import List
from financialmath.pricing.futures.schema import (
    FutureValuationFunction, FutureInputData)
from financialmath.instruments.futures import Future, FutureType
from financialmath.tools.tool import MainTool

@dataclass
class FuturePremium: 

    @staticmethod
    def equity(data: FutureInputData) -> float:
        return data.interest_rate_domestic - data.dividend_yield
    
    @staticmethod
    def fx(data: FutureInputData) -> float:
        rd = data.interest_rate_domestic 
        rf = data.interest_rate_foreign
        bs = data.basis_spread
        return rd - rf + bs 
    
    @staticmethod
    def commodity(data: FutureInputData) -> float:
        r = data.interest_rate_domestic
        u = data.storage_cost
        y = data.covenience_yield
        return r+u-y
    
    @staticmethod
    def crypto(data: FutureInputData) -> float:
        return data.interest_rate_domestic

class FuturePremiumList(Enum): 

    equity = FuturePremium.equity
    fx = FuturePremium.fx
    commodity = FuturePremium.commodity
    crypto = FuturePremium.crypto

    def get_premium_method(self, future_type:FutureType): 
        try: 
            premlist = list(FuturePremiumList)
            return [m.value for m in premlist
                    if m.name == future_type.name][0]
        except Exception as e: return None 

@dataclass
class FuturePricing(FutureValuationFunction): 
    S : float or List[float]
    x : float or List[float]
    t : float or List[float]
    continuous : bool 

    def __post_init__(self):
        self.S = MainTool.convert_to_numpy_array(x=self.S)
        self.x = MainTool.convert_to_numpy_array(x=self.x)
        self.t = MainTool.convert_to_numpy_array(x=self.t)
    
    def method(self) -> str: 
        if self.continuous: return 'Future formula w/ continuous compounding'
        else: 'Future formula w/ discrete compounding'

    def delta(self) -> np.array: 
        if self.continuous: return np.exp(self.x*self.t)
        else: return (1+self.x)**self.t
    
    def price(self) -> np.array: 
        return self.S*self.delta()

    def theta(self) -> np.array:
        if self.continuous: return self.x*self.price()
        else: return np.log(1+self.x)*self.price()
    
    def rho(self) -> np.array: 
        if self.continuous: return self.t*self.price()
        else: return self.S*self.t*((1+self.x)**(self.t-1))

@dataclass
class InterestRateFuturePricing(FutureValuationFunction):

    r : float or List[float]

    def __post_init__(self):
        self.r = MainTool.convert_to_numpy_array(x=self.S)
        if not isinstance(self.r, float): self.n = len(self.r)
    
    def method(self) -> str: 
        return 'Interest rate Future method'

    def price(self) -> float or np.array: 
        return 100*(1-self.r)
    
    def rho(self) -> float or np.array: 
        return -100*self.r
    
    def theta(self)-> float or np.array:
        if not isinstance(self.r, float): return np.repeat(np.nan, self.n)
        else: return np.nan
    
    def delta(self)-> float or np.array:
        if not isinstance(self.r, float): return np.repeat(np.nan, self.n)
        else: return np.nan
    
@dataclass
class BondFuturePricing(FutureValuationFunction): 

    n : int

    def method(self) -> str: 
        return 'Bond Future method not developed'

    def price(self) -> float or np.array: 
        if self.n>1: return np.repeat(np.nan, self.n)
        else: return np.nan
    
    def rho(self) -> float or np.array: 
        if self.n>1: return np.repeat(np.nan, self.n)
        else: return np.nan
    
    def theta(self)-> float or np.array:
        if self.n>1: return np.repeat(np.nan, self.n)
        else: return np.nan
    
    def delta(self)-> float or np.array:
        if self.n>1: return np.repeat(np.nan, self.n)
        else: return np.nan