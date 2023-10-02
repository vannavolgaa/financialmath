from dataclasses import dataclass
from financialmath.instruments.option import Option
from abc import abstractmethod, ABC
from enum import Enum 
import numpy as np

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
class OptionValuationResult: 
    instrument : Option
    price : float 
    sensitivities : OptionGreeks
    method : str
    marketdata : ImpliedOptionMarketData





