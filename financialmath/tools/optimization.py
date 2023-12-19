from scipy import optimize, sparse
import numpy as np 
from dataclasses import dataclass

@dataclass
class NewtonRaphsonMethod: 
    
    f : object 
    df : object 
    x_0 : float or np.array 
    epsilon : float 
    max_iterations : int = 100

    def fall_back(self) -> np.array: 
        if isinstance(self.x_0, float): return np.nan
        else: 
            n = len(self.x_0)
            return np.repeat(np.nan,n)
    
    def find_x(self): 
        x = self.x_0
        error = np.abs(self.f(x))
        i = 0
        while np.all(error>self.epsilon): 
            x = x - self.f(x)/self.df(x)
            error = np.abs(self.f(x))
            i = i + 1
            if i == self.max_iterations:
                return self.fall_back()
        return x
            
            
        