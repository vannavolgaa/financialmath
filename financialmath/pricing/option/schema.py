from dataclasses import dataclass
from abc import abstractmethod, ABC
from enum import Enum 
import numpy as np
from typing import List
from financialmath.quanttool import QuantTool
from financialmath.instruments.option import Option

@dataclass
class ImpliedOptionMarketData: 
    S : float 
    r : float 
    q : float 
    sigma : float 
    F : float

@dataclass
class OptionGreeks: 
    delta: float = np.nan
    vega: float = np.nan
    theta: float = np.nan
    rho: float = np.nan
    epsilon: float = np.nan
    gamma: float = np.nan
    vanna: float = np.nan
    volga: float = np.nan
    charm: float = np.nan
    veta: float = np.nan
    vera: float = np.nan
    speed: float = np.nan
    zomma: float = np.nan
    color: float = np.nan
    ultima: float = np.nan

@dataclass
class OptionValuationResult: 
    instrument : Option
    marketdata : ImpliedOptionMarketData
    price : float 
    sensitivities : OptionGreeks
    method : str
    

class OptionValuationFunction(ABC): 

    greek_name = ['delta', 'vega', 'gamma', 'rho', 'epsilon', 'theta', 
                  'vanna', 'volga', 'speed', 'charm', 'veta', 'vera', 
                  'zomma', 'ultima', 'color']

    def get_price(self,n:int) -> List[float]: 
        try: return QuantTool.convert_array_to_list(self.price())
        except Exception as e: return [np.nan]*n
    
    def get_method(self, n:int) -> List[str]: 
        try: return [self.method()]*n
        except Exception as e: return [None]*n
    
    def get_greeks(self, n:int) -> List[OptionGreeks]: 
        try:
            greeks = [self.delta(), self.vega(), self.gamma(), self.rho(), 
                  self.epsilon(), self.theta(), self.vanna(), self.volga(), 
                  self.speed(), self.charm(), self.veta(), self.vera(), 
                  self.zomma(), self.ultima(), self.color()]
            data = {n:QuantTool.convert_array_to_list(d) 
                        for n,d in zip(self.greek_name, greeks)}
            data = QuantTool.dictlist_to_listdict(data)
            return [OptionGreeks(**d) for d in data]
        except Exception as e: return [OptionGreeks()]*n
        
    @abstractmethod
    def method(self)  -> float or np.array: pass

    @abstractmethod
    def price(self) -> float or np.array: pass

    @abstractmethod
    def delta(self) -> float or np.array: pass 

    @abstractmethod
    def vega(self) -> float or np.array: pass 

    @abstractmethod
    def gamma(self) -> float or np.array: pass 

    @abstractmethod
    def rho(self) -> float or np.array: pass 

    @abstractmethod
    def theta(self) -> float or np.array: pass 

    @abstractmethod
    def epsilon(self) -> float or np.array: pass 

    @abstractmethod
    def vanna(self) -> float or np.array: pass 

    @abstractmethod
    def volga(self) -> float or np.array: pass

    @abstractmethod
    def speed(self) -> float or np.array: pass 

    @abstractmethod
    def charm(self) -> float or np.array: pass 

    @abstractmethod
    def veta(self) -> float or np.array: pass

    @abstractmethod 
    def vera(self) -> float or np.array: pass

    @abstractmethod 
    def zomma(self) -> float or np.array: pass

    @abstractmethod 
    def ultima(self) -> float or np.array: pass

    @abstractmethod 
    def color(self) -> float or np.array: pass 


    
    




