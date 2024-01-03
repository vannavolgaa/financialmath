from dataclasses import dataclass
import numpy as np 
from typing import List
from financialmath.instruments.option import MarketOptionQuotes

@dataclass
class EuropeanPutCallParityInput: 
    call : MarketOptionQuotes
    put : MarketOptionQuotes
    S_bid : float 
    S_ask : float 
    r_bid : float = np.nan
    r_ask : float = np.nan
    
    def __post_init__(self): 
        self.S_mid = (self.S_ask+self.S_bid)/2
        self.r_mid = (self.r_ask+self.r_bid)/2
        self.is_r_nan = np.isnan(self.r_mid)

@dataclass
class EuropeanPutCallParity: 
    inputdata : List[EuropeanPutCallParityInput]
    bid_ask : bool = True






