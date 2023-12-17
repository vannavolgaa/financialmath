from abc import ABC, abstractmethod
import numpy as np 

class VolatilitySurface(ABC): 

    @abstractmethod
    def implied_variance(self, k:np.array, t: np.array) -> np.array: 
        pass 

    @abstractmethod
    def total_variance(self, k:np.array, t: np.array) -> np.array: 
        pass 

    @abstractmethod
    def implied_volatility(self, k:np.array, t: np.array) -> np.array: 
        pass 

    @abstractmethod
    def local_volatility(self, k:np.array, t: np.array) -> np.array: 
        pass 

    @abstractmethod
    def skew(self, k:np.array, t:np.array) -> np.array: 
        pass 