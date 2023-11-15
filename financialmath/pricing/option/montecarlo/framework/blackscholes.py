from dataclasses import dataclass
from financialmath.tools.probability import NormalDistribution
import numpy as np 
from enum import Enum 

class PathType(Enum): 
    moneyness = 1 
    price = 2 
    log_moneyness = 3

@dataclass
class BlackScholesDiffusion: 
    
    r: float 
    q: float 
    sigma : float
    dt : float 
    Z : np.array
    sensitivities : bool = True

    def drift(self, r: float, q:float,sigma:float) -> float: 
        return (r-q-0.5*(sigma**2))*self.dt
    
    def diffusion(self, sigma: float) -> np.array: 
        return sigma*np.sqrt(self.dt)*self.Z 
    
    def milstein_correction(self, sigma:float) -> np.array: 
        return .5*(sigma**2)*((self.Z)**2 - 1)*self.dt
    
    def generate_euler_moneyness_path(self, r: float, q:float, 
                                    sigma:float) -> np.array: 
        mu = self.drift(r, q, sigma)
        diffusion = self.diffusion(sigma)
        return np.cumprod(np.exp(mu+diffusion), axis = 1)

    def generate_euler_price_path(self, r: float, S:float, q:float, 
                                    sigma:float) -> np.array: 
        return S*self.generate_euler_moneyness_path(r=r, q=q, sigma=sigma)

    def generate_logmoneyness_path(self, r: float, q:float, 
                                    sigma:float) -> np.array: 
        mu = self.drift(r, q, sigma)
        diffusion = self.diffusion(sigma)
        return np.cumsum(mu+diffusion, axis = 1)
    
    def generate_milstein_price_path(self,S:float, r: float, q:float, 
                                    sigma:float) -> np.array: 
        euler = self.generate_euler_moneyness_path(r=r, q=q, sigma=sigma)
        correction = self.milstein_correction(sigma=sigma)
        return S*euler+correction



@dataclass
class SimulationObject: 
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
