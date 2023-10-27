from dataclasses import dataclass
from abc import abstractmethod, ABC
from financialmath.instruments.futures import Future

class FutureValuationFunction(ABC): 

    @abstractmethod
    def price(self): 
        pass 

    @abstractmethod
    def delta(self): 
        pass 

    @abstractmethod
    def theta(self): 
        pass 

    @abstractmethod
    def rho(self): 
        pass 

@dataclass
class FutureInputData: 
    spot : float = np.nan
    interest_rate_domestic : float = np.nan 
    interest_rate_foreign : float = np.nan 
    basis_spread : float = np.nan 
    dividend_yield : float = np.nan
    storage_cost : float = np.nan 
    covenience_yield : float = np.nan 

@dataclass
class FutureSensibility: 
    delta : float = np.nan 
    theta : float = np.nan 
    rho : float = np.nan 

@dataclass
class FutureValuationResult: 
    instrument : Future
    marketdata : FutureInputData
    price : float 
    sensitivities : FutureSensibility
    method : str


