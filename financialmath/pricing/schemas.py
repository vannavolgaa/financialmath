from dataclasses import dataclass
import numpy as np
from financialmath.instruments.option import Option 

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


@dataclass
class OptionValuationResult: 
    instrument : Option
    inputdata : classmethod
    price : float 
    sensitivities : OptionGreeks
    method : str
    time_taken : float 
    