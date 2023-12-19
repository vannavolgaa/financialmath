import numpy as np
from dataclasses import dataclass
from financialmath.tools.probability import NormalDistribution
from financialmath.tools.optimization import NewtonRaphsonMethod
from financialmath.model.blackscholes.closedform import (
    BlackEuropeanVanillaCall, 
    BlackEuropeanVanillaPut, 
    BlackScholesEuropeanVanillaCall, 
    BlackScholesEuropeanVanillaPut, 
    ClosedFormBlackScholesInput
)

class QuadraticApproximationAmericanVanillaCall: 
    
    def __init__(self, inputdata:ClosedFormBlackScholesInput, 
                 future:bool = False):
        self.future = future
        self.S, self.K, self.q, self.r, self.t, self.sigma = inputdata.S,\
             inputdata.K,inputdata.q,inputdata.r,inputdata.t,inputdata.sigma
        if future: self.b = 0 
        else: self.b = self.r-self.q 
        self.bs = self.return_black_scholes(inputdata)
        self.F = 1-np.exp(-self.r*self.t)
        self.N = 2*self.b/self.sigma**2
        self.M = 2*self.r/self.sigma**2
        self.q2 = (-(self.N-1)-np.sqrt((self.N-1)**2+4*self.M/self.F))/2
        self.S_star = self.find_optimal_exercise_price()
        self.euro_prices = self.bs.price()
    
    def return_black_scholes(self, inputdata: ClosedFormBlackScholesInput): 
        if self.future: return BlackEuropeanVanillaCall(
                inputdata=inputdata) 
        else: return BlackScholesEuropeanVanillaCall(
                inputdata=inputdata)
    
    def rhs(self, S:np.array) -> np.array: 
        inputdata = ClosedFormBlackScholesInput(
            S = S, r = self.r, q = self.q, 
            sigma = self.sigma, t = self.t, K = self.K
        )
        bs = self.return_black_scholes(inputdata=inputdata)
        return bs.price() + S*(1-bs.delta())/self.q2 
     
    def minimize_function(self, S: np.array) -> np.array: 
        return np.abs(S-self.K-self.rhs(S=S))/self.K 

    def rhs_slope(self, S:np.array) -> np.array: 
        inputdata = ClosedFormBlackScholesInput(
            S = S, r = self.r, q = self.q, 
            sigma = self.sigma, t = self.t, K = self.K
        )
        bs = self.return_black_scholes(inputdata=inputdata)
        delta, gamma = bs.delta(), bs.gamma()
        return delta + (1-delta+S*gamma)/self.q2

    def find_optimal_exercise_price(self) -> np.array:
        S = self.K * 1.1     
        error = self.minimize_function(S=S)       
        while error>0.000001: 
            b = self.rhs_slope(S=S)
            S = (self.K + self.rhs(S=S)-b*S)/(1-b)
            error=self.minimize_function(S=S)
        return S
    
    def early_exercise(self) -> np.array: 
        return self.S - self.K
    
    def exercise_premium(self) -> np.array: 
        inputdata = ClosedFormBlackScholesInput(
            S = self.S_star, 
            r = self.r, 
            q = self.q, 
            sigma = self.sigma, 
            t = self.t, 
            K = self.K
        )
        bs = self.return_black_scholes(inputdata=inputdata)
        delta = bs.delta()
        factor = self.S_star*((self.S/self.S_star)**self.q2)
        return factor*(1-delta)/self.q2
    
    def compute_prices(self) -> np.array: 
        early_ex = self.early_exercise()
        exercise_premium = self.exercise_premium()
        if isinstance(self.S_star, float):
            if self.S>=self.S_star: result = early_ex
            else: result = self.euro_prices + exercise_premium
        else: 
            indexes = np.where(self.S>=self.S_star)[0]
            inv_indexes =  np.where(self.S<self.S_star)[0]
            result = np.zeros(self.S_star.shape)
            result[indexes] = early_ex[indexes]
            result[inv_indexes] = np.maximum(exercise_premium[inv_indexes],0)
        return result 
        

