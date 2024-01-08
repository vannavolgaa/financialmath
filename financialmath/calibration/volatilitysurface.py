import numpy as np 
from dataclasses import dataclass
from scipy.optimize import least_squares
from typing import List
from py_lets_be_rational import implied_volatility_from_a_transformed_rational_guess
from financialmath.model.parametricvolatility import ParametricVolatility
from financialmath.instruments.option import (
    MarketOptionQuotes, 
    OptionalityType
)
from financialmath.instruments.spot import MarketSpotQuotes
from financialmath.marketdata.termstructure import YieldCurve, TermStructure

@dataclass
class ParametricVolatilityCalibrationFromEuropeanPrices: 
    option_quotes : List[MarketOptionQuotes]
    spot_quote : MarketSpotQuotes
    yield_curve : YieldCurve
    fit_carry_cost : bool = True 

    def __post_init__(self): 
        pass

    def coefficients(self, iv: np.array, k:np.array) -> np.array: 
        lk = np.log(self.k)
        lks = lk**2
        n = len(k)
        xlist = [np.ones(n), lk, lks]
        Xt = np.reshape(np.concatenate(xlist),(len(xlist),n))
        X = np.transpose(Xt)
        Y = iv
        return np.linalg.lstsq(X, Y, rcond=None)[0]
    
    
    

        
