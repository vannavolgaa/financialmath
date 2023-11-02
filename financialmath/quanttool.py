import numpy as np 
from typing import List

class QuantTool: 

    @staticmethod
    def dictlist_to_listdict(mydict: dict): 
        keys = list(mydict.keys())
        length = len(mydict[keys[0]])
        output = []
        for i in range(0, length): 
            output.append({k : mydict[k][i] for k in keys})
        return output 

    
    @staticmethod
    def convert_to_numpy_array(x:float or List[float]): 
        if isinstance(x, list): return np.array(x)
        else: return x
    
    @staticmethod
    def compute_blackscholes_d1(S, K, r, q, t, sigma):
        return(np.log(S/K)+(r-q+sigma**2/2)*t)/(sigma*np.sqrt(t))
    
    @staticmethod
    def compute_blackscholes_d2(S, K, r, q, t, sigma):
        d1 = QuantTool.compute_blackscholes_d1(S=S,K=K,t=t,r=r,q=q, sigma=sigma)
        return d1-sigma*np.sqrt(t)
    
    @staticmethod
    def convert_array_to_list(x: float or np.array): 
        if isinstance(x, float): return [x]
        else: return list(x)
    
    @staticmethod
    def convert_any_to_list(x): 
        if not isinstance(x, list): return [x]
        else: return x





