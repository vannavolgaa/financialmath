import numpy as np 
from dataclasses import dataclass
from scipy.optimize import least_squares
from typing import List
from financialmath.model.parametricvolatility import ParametricVolatility
from financialmath.instruments.option import MarketOptionQuotes
from financialmath.instruments.spot import MarketSpotQuotes
from financialmath.marketdata.schemas import YieldCurve

@dataclass
class LeastSquareRegressionParametricVolatility: 
    ivs : np.array 
    k : np.array 
    t : np.array 
    smile : bool = True 
    
    def x_matrix(self): 
        lk = np.log(self.k)
        lks = lk**2
        tlk = self.t*lk 
        n = len(self.ivs)
        if self.smile: xlist = [np.ones(n), lk, lks]
        else: xlist = [np.ones(n), lk, lks, self.t, tlk]
        return np.transpose(np.reshape(np.concatenate(xlist),(len(xlist),n)))
    
    def coefficients(self) -> np.array: 
        X = self.x_matrix()
        Y = self.ivs
        return np.linalg.lstsq(X, Y, rcond=None)[0]

@dataclass
class SSVICalibrationFromVanillaPrices: 
    options : List[MarketOptionQuotes]
    spot : MarketSpotQuotes
    yield_curve : YieldCurve
    fit_carry_cost : bool = True 

    
        
    
    
        
    
    
        
