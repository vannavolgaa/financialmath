from scipy import optimize, sparse
import numpy as np 
from dataclasses import dataclass

@dataclass
class NewtonRaphsonMethod: 
    
    f : object 
    df : object 
    x_0 : float or np.array 
    epsilon : float 
    c : object
    
    def find_x(self): 
        x = self.x_0
        error = np.abs(self.f(x))
        while np.all(error>self.epsilon): 
            x = x - self.f(x)/self.df(x)
            error = np.abs(self.f(x))
            if self.c is not None: 
                error = error + self.c(x)
        return x
            
            
        