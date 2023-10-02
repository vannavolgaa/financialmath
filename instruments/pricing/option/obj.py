from dataclasses import dataclass
from instruments.option import Option
from abc import abstractmethod, ABC
from enum import Enum 

@dataclass
class OptionGreeks: 
    delta: float 
    vega: float 
    theta: float 
    rho: float 
    epsilon: float 
    gamma: float 
    vanna: float 
    volga: float 
    charm: float 
    veta: float 
    vera: float 
    speed: float 
    zomma: float 
    color: float 
    ultima: float 

class OptionValuationFunction(ABC): 
    @abstractmethod
    def method(self)  -> float or np.array: 
        pass
    @abstractmethod
    def price(self) -> float or np.array: 
        pass 
    @abstractmethod
    def delta(self) -> float or np.array: 
        pass 
    @abstractmethod
    def vega(self) -> float or np.array: 
        pass 
    @abstractmethod
    def gamma(self) -> float or np.array: 
        pass 
    @abstractmethod
    def rho(self) -> float or np.array: 
        pass 
    @abstractmethod
    def theta(self) -> float or np.array: 
        pass 
    @abstractmethod
    def epsilon(self) -> float or np.array: 
        pass 
    @abstractmethod
    def vanna(self) -> float or np.array: 
        pass 
    @abstractmethod
    def volga(self) -> float or np.array: 
        pass 
    @abstractmethod
    def speed(self) -> float or np.array: 
        pass 
    @abstractmethod
    def charm(self) -> float or np.array: 
        pass 
    @abstractmethod
    def veta(self) -> float or np.array: 
        pass
    @abstractmethod 
    def vera(self) -> float or np.array: 
        pass
    @abstractmethod 
    def zomma(self) -> float or np.array: 
        pass
    @abstractmethod 
    def ultima(self) -> float or np.array: 
        pass 
    @abstractmethod 
    def color(self) -> float or np.array: 
        pass 

class ImpliedOptionMarketData: 
    S : float 
    r : float 
    q : float 
    sigma : float 

@dataclass
class OptionSummary: 
    instrument : Option
    price : float 
    sensitivities : OptionGreeks
    method : str
    marketdata : ImpliedOptionMarketData





