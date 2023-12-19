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
                 future:bool = False, put:bool = False):
        self.future, self.put = future, put
        self.S, self.K, self.q, self.r, self.t, self.sigma = inputdata.S,\
             inputdata.K,inputdata.q,inputdata.r,inputdata.t,inputdata.sigma
        if future: self.b = 0 
        else: self.b = self.r-self.q 
        self.bs = self.return_black_scholes(inputdata)
        self.F = 1-np.exp(-self.r*self.t)
        self.N = 2*self.b/self.sigma**2
        self.M = 2*self.r/self.sigma**2
        self.q2 = (-(self.N-1)+np.sqrt((self.N-1)**2+4*self.M/self.F))/2
        self.q1 = (-(self.N-1)-np.sqrt((self.N-1)**2+4*self.M/self.F))/2
        self.S_star = self.find_optimal_exercise_price()
        self.euro_prices = self.bs.price()
    
    def return_black_scholes(self, inputdata: ClosedFormBlackScholesInput):
        if self.future: 
            if self.put: return BlackEuropeanVanillaPut(inputdata)
            else: return BlackEuropeanVanillaCall(inputdata) 
        else: 
            if self.put: return BlackScholesEuropeanVanillaPut(inputdata)
            else: return BlackScholesEuropeanVanillaCall(inputdata) 
    
    def minimize_function(self, S: np.array) -> np.array:
        inputdata = ClosedFormBlackScholesInput(
            S = S, r = self.r, q = self.q, 
            sigma = self.sigma, t = self.t, K = self.K
        )
        bs = self.return_black_scholes(inputdata=inputdata)
        price, delta = bs.price(), bs.delta()
        if self.put: return self.K - S - price + S*(1+delta)/self.q1
        else: return S - self.K - price - S*(1-delta)/self.q2
    
    def minimize_function_derivative(self, S: np.array) -> np.array:
        inputdata = ClosedFormBlackScholesInput(
            S = S, r = self.r, q = self.q, 
            sigma = self.sigma, t = self.t, K = self.K
        )
        bs = self.return_black_scholes(inputdata=inputdata)
        gamma = bs.gamma()
        if self.put: return (1/self.q1)-1+S*gamma
        else: return 1-(1/self.q2)+S*gamma
    
    def find_optimal_exercise_price(self) -> np.array:
        if self.put: S_0 = self.K*0.9
        else: S_0 = self.K*1.1 
        return NewtonRaphsonMethod(
            f = self.minimize_function,
            df = self.minimize_function_derivative, 
            x_0 = S_0, 
            epsilon=0.00001
        ).find_x()

    def early_exercise(self) -> np.array: 
        if self.put: return self.K - self.S
        else: return self.S - self.K
    
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
        if self.put: 
            factor = self.S_star*((self.S/self.S_star)**self.q1)
            return factor*(1+delta)/self.q1
        else: 
            factor = self.S_star*((self.S/self.S_star)**self.q2)
            return factor*(1-delta)/self.q2
    
    def vectorized_compute_prices(self, early_ex:np.array, 
                                  exprem:np.array, 
                                  early_ex_cond: np.array) -> np.array: 
        indexes = np.where(early_ex_cond)[0]
        inv_cond = np.logical_not(early_ex_cond)
        inv_indexes =  np.where(inv_cond)[0]
        nan_index = np.where(np.isnan(self.S_star))[0]
        result = np.zeros(self.S_star.shape)
        try : result[indexes] = early_ex[indexes]
        except TypeError: 
            early_ex = np.repeat(early_ex, len(self.S_star))
            result[indexes] = early_ex[indexes]
        result[inv_indexes] = self.euro_prices + exprem[inv_indexes]
        result[nan_index] = np.repeat(self.euro_prices, len(nan_index))
        return result
    
    def early_exercise_cond(self) -> np.array: 
        if self.put: return (self.S<=self.S_star)
        else: return (self.S>=self.S_star)

    def compute_prices(self) -> np.array: 
        early_ex = self.early_exercise()
        exercise_premium = self.exercise_premium()
        early_exercise_cond = self.early_exercise_cond()
        if isinstance(self.S_star, float):
            if early_exercise_cond: result = early_ex
            else: 
                if np.isnan(exercise_premium): exercise_premium=0
                result = self.euro_prices + exercise_premium
        else: 
            result = self.vectorized_compute_prices(
                early_ex=early_ex, 
                exprem=exercise_premium,
                early_ex_cond=early_exercise_cond
            )
        return result 
        

