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
    def risk_neutral_density(self, k:np.array, t:np.array) -> np.array: 
        pass 

class YieldCurve(ABC): 

    @abstractmethod
    def rate(self, t:np.array) -> np.array: 
        pass 

    @abstractmethod
    def forward_rate(self, t:np.array, T:np.array) -> np.array: 
        pass 

    def continuous_discount_factor(self, t:np.array) -> np.array: 
        zr = self.rate(t=t)
        return np.exp(-zr*t)

class TermStructure(ABC): 

    @abstractmethod
    def rate(self, t:np.array) -> np.array: 
        pass