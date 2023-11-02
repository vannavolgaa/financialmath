from dataclasses import dataclass
from abc import abstractmethod, ABC
from financialmath.instruments.futures import Future
import math
from typing import List
from financialmath.quanttool import QuantTool

class FutureValuationFunction(ABC): 

    def sensi(self) -> List[dict]: 
        name = ['delta', 'theta', 'rho']
        sensi = [self.delta(), self.theta(), 
                self.rho()]
        data = {n:QuantTool.convert_array_to_list(d) 
                    for n,d in zip(name, sensi)}
        data = QuantTool.dictlist_to_listdict(data)
        return data

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
    spot : float = math.nan
    interest_rate_domestic : float = math.nan 
    interest_rate_foreign : float = math.nan 
    basis_spread : float = math.nan 
    dividend_yield : float = math.nan
    storage_cost : float = math.nan 
    covenience_yield : float = math.nan 

@dataclass
class FutureSensibility: 
    delta : float = math.nan 
    theta : float = math.nan 
    rho : float = math.nan 

@dataclass
class FutureValuationResult: 
    instrument : Future
    marketdata : FutureInputData
    price : float 
    sensitivities : FutureSensibility
    method : str


