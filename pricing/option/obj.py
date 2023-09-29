from dataclasses import dataclass
from instruments.option import Option
from abc import abstractmethod, ABC

@dataclass
class OptionValuation: 
    price: float 
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
    def price(self) -> float: 
        pass 
    @abstractmethod
    def delta(self) -> float: 
        pass 
    @abstractmethod
    def vega(self) -> float: 
        pass 
    @abstractmethod
    def gamma(self) -> float: 
        pass 
    @abstractmethod
    def rho(self) -> float: 
        pass 
    @abstractmethod
    def epsilon(self) -> float: 
        pass 
    @abstractmethod
    def vanna(self) -> float: 
        pass 
    @abstractmethod
    def volga(self) -> float: 
        pass 
    @abstractmethod
    def speed(self) -> float: 
        pass 
    @abstractmethod
    def charm(self) -> float: 
        pass 
    @abstractmethod
    def veta(self) -> float: 
        pass
    @abstractmethod 
    def vera(self) -> float: 
        pass
    @abstractmethod 
    def zomma(self) -> float: 
        pass
    @abstractmethod 
    def ultima(self) -> float: 
        pass 
    @abstractmethod
    def main(self) -> OptionValuation: 
        pass 

@dataclass
class OptionSummary: 
    instrument : Option
    valuation : OptionValuation
    method : str



