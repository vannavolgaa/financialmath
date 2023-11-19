from dataclasses import dataclass
from financialmath.tools.probability import NormalDistribution
import numpy as np 
from enum import Enum 

@dataclass
class EulerBlackScholesMoneyness: 
    
    r: float 
    q: float 
    sigma : float
    dt : float 
    Z : np.array


    def drift(self, r: float, q:float,sigma:float) -> float: 
        return (r-q-0.5*(sigma**2))*self.dt
    
    def diffusion(self, sigma: float) -> np.array: 
        return sigma*np.sqrt(self.dt)*self.Z 
    
    def generate_path(self, r: float, q:float, 
                                    sigma:float) -> np.array: 
        mu = self.drift(r, q, sigma)
        diffusion = self.diffusion(sigma)
        return np.cumprod(np.exp(mu+diffusion), axis = 1)
    

    





@dataclass
class EulerSimulationObject: 
    initial : np.array 
    vol_up : np.array = None
    vol_down : np.array = None
    r_up : np.array = None
    q_up : np.array = None
    dt : float = 0
    spot_bump_size: float = 0.01
    volatility_bump_size: float = 0.01
    r_bump_size : float = 0.01
    q_bump_size : float = 0.01

