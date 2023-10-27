from dataclasses import dataclass
from financialmath.pricing.futures.schema import FutureValuationFunction
from financialmath.instruments.futures import Future
import numpy as np 

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
            return [m.value for m in list(MappingPremium) 
                    if m.name == future_type.name][0]
        except Exception as e: return None 

@dataclass
class FuturePricing(FutureValuationFunction): 
    S : float or List[float]
    x : float or List[float]
    t : float or List[float]
    continuous : bool 

    def __post_init__(self):
        self.S = QuantTool.convert_to_numpy_array(x=self.S)
        self.x = QuantTool.convert_to_numpy_array(x=self.x)
        self.t = QuantTool.convert_to_numpy_array(x=self.t)
    
    def method(self) -> str: 
        if self.continuous: return 'Future formula w/ continuous compounding'
        else: 'Future formula w/ discrete compounding'

    def delta(self): 
        if self.continuous: return np.exp(self.x*self.t)
        else: return (1+self.x)**self.t
    
    def price(self): 
        return self.S*self.delta()

    def theta(self):
        if self.continuous: return self.x*self.price()
        else: return np.log(1+self.x)*self.price()
    
    def rho(self): 
        if self.continuous: return self.t*self.price()
        else: return self.S*self.t*((1+self.x)**(t-1))

@dataclass
class InterestRateFuturePricing(FutureValuationFunction): 
    r : float or List[float]
    def __post_init__(self):
        self.r = QuantTool.convert_to_numpy_array(x=self.S)
        if not isinstance(self.r, float): self.n = len(r)
    
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
    def method(self) -> str: 
        return 'Bond Future method'

    def price(self) -> float or np.array: 
        return np.nan
    
    def rho(self) -> float or np.array: 
        return np.nan
    
    def theta(self)-> float or np.array:
        return np.nan
    
    def delta(self)-> float or np.array:
        return np.nan