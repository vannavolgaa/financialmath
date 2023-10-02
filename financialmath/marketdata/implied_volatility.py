from dataclasses import dataclass
from typing import List

@dataclass
class ImpliedVolatilityPoint: 
    t : float 
    strike : float 
    moneyness : float 
    delta : float 
    volatility : float 

@dataclass
class ImpliedVolatilitySmile: 
    t : float 
    points : List[ImpliedVolatilityPoint]

@dataclass
class ImpliedVolatilitySurface: 
    smiles : ImpliedVolatilitySmile
