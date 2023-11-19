from dataclasses import dataclass
from typing import List
import numpy as np
from enum import Enum
from financialmath.instruments.option import MarketOptionQuotes
from financialmath.instruments.futures import MarketFutureQuotes
from financialmath.instruments.spot import MarketSpotQuotes
from financialmath.instruments.zcbond import MarketZCBondQuotes

class ImpliedQuoteTypes(Enum): 
    yields = 1
    implied_volatility = 2 

@dataclass
class MarketQuotes: 
    option : MarketOptionQuotes or List[MarketOptionQuotes] = None
    spot : MarketSpotQuotes = None
    domestic_zcbond : MarketZCBondQuotes = None
    foreign_zcbond : MarketZCBondQuotes = None
    future : MarketFutureQuotes = None

@dataclass
class ImpliedQuotes: 
    bid : float 
    ask : float 
    expiry : float
    data_types : ImpliedQuoteTypes
    market_quote : MarketQuotes
    
    def __post_init__(self): 
        self.mid = (self.bid + self.ask)/2

class VolatilityType(Enum): 
    local = 1 
    implied_volatility = 2
    implied_variance = 3
    implied_total_variance = 4 

class MoneynessType(Enum): 
    strike = 1 
    delta = 2 
    moneyness = 3 
    log_moneyness = 4 
    log_forward_moneyness = 5 

@dataclass
class TermPoint: 
    t : float 
    value : float 

@dataclass
class TermStructure: 
    points : List[TermPoint]

@dataclass
class OptionVolatilityPoint: 
    t : float 
    moneyness_type : MoneynessType
    moneyness : float 
    volatility_type : VolatilityType
    sigma : float  

@dataclass
class OptionVolatilitySurface: 
    points : List[OptionVolatilityPoint]

    def volatility_matrix(self) -> np.array: 
        output = []
        t_list=sorted(list(set([s.t for s in self.points])))
        N = len(t_list)
        for t in t_list: 
            data = {s.moneyness : s.sigma for s in self.points if s.t == t}
            data = {k : data[k] for k in sorted(data)}
            output = output + list(data.values())
        M = int(len(output)/N)
        try:return np.reshape(output, (M,N))
        except Exception as e: return np.reshape(np.repeat(np.nan, M*N), (M,N))



