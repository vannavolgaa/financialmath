from dataclasses import dataclass
from typing import List

@dataclass
class OptionVolatilityPoint: 
    t : float 
    moneyness: float 
    sigma : float 

@dataclass
class OptionVolatilitySurface: 
    points : List[OptionVolatilityPoint]
