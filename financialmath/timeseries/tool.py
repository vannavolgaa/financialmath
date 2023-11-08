from enum import Enum
from dataclasses import dataclass
import numpy as np
from timeseries.schemas import TimeSerieDefinition, TimeIntervalDefinition

@dataclass
class ComputeReturns: 

    data : TimeSerieDefinition

    @staticmethod
    def compute_simple_difference(X: np.array) -> np.array: 
        return np.diff(X)

    @staticmethod
    def compute_log_return(X: np.array) -> np.array: 

        return np.diff(np.log(X))

    def compute_simple_return(self, X: np.array) -> np.array: 

        X_shift = np.delete(X, len(X)-1)

        return self.compute_simple_difference(X=X)/X_shift

class RealisedVolatility:
    pass

    

