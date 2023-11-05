from dataclasses import dataclass
from typing import List
import numpy as np
from enum import Enum

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
        return np.reshape(output, (M,N))



