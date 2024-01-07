from abc import ABC, abstractmethod
from enum import Enum 
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np 
from typing import List
from scipy import interpolate
from financialmath.marketdata.schemas import TermStructure
from financialmath.tools.tool import MainTool

@dataclass
class ExtrapolatedTotalVarianceTermStructure(TermStructure): 
    t : List[float]
    totvariance : List[float]
    max_t : int = 100

    interpolation_method = 'cubic'

    def __post_init__(self): 
        ordered_dict = MainTool.order_join_lists(
            keys=self.t, 
            values=self.totvariance)
        self.t = np.array(list(ordered_dict.keys()))
        self.totvariance = np.array(list(ordered_dict.values()))
        self.extrapolator = self.extrapolation()

    def extrapolation(self) -> interpolate.interp1d: 
        tvar, t = self.totvariance, self.t
        n = len(self.tvar)
        tvar2, t2, tvar1, t1 = tvar[n-1], t[n-1], tvar[n-2], t[n-2]
        slope = (tvar2-tvar1)/(t2-t1)
        max_tvar = slope*self.max_t
        tvar = np.insert(tvar, [0, len(tvar)], [0, max_tvar])
        t = np.insert(t, [0, len(t)], [0, self.max_t])
        return interpolate.interp1d(t, tvar, kind = self.interpolation_method)
    
    def rate(self, t: np.array) -> np.array: 
        return self.extrapolator(x=t)