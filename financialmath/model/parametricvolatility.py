from dataclasses import dataclass
import numpy as np 

@dataclass
class ParametricVolatility: 
    b0 : float 
    b1 : float = 0
    b2 : float = 0
    b3 : float = 0
    b4 : float = 0 
    
    def implied_volatility(self, k: np.array, t: np.array) -> np.array: 
        b0, b1, b2, b3, b4 = self.b0, self.b1, self.b2, self.b3, self.b4
        lk = np.log(k)
        return b0 + b1*lk + b2*(lk**2) + b3*t + b4*t*lk
    

