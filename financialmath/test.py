from financialmath.model.blackscholes.montecarlo import (BlackScholesDiscretization, 
MonteCarloBlackScholes, MonteCarloBlackScholesInput)
from financialmath.model.blackscholes.pde import (PDEBlackScholes, PDEBlackScholesInput)
import numpy as np 
import matplotlib.pyplot as plt 
from financialmath.instruments.option import *
from dataclasses import dataclass
from financialmath.pricing.option.pde import PDEBlackScholesValuation
import matplotlib.pyplot as plt
from financialmath.pricing.numericalpricing.option import MonteCarloPricing
import time
from enum import Enum

lookback_payoff_floatK = LookBackPayoff(floating_strike=True, floating_spot=False, 
                                 spot_method=None, strike_method=LookbackMethod.geometric_mean, 
                                 spot_observation=None, strike_observation=ObservationType.continuous)
lookback_payoff_floatS = LookBackPayoff(floating_strike=False, floating_spot=True, 
                                 spot_method=LookbackMethod.geometric_mean, strike_method=None, 
                                 spot_observation=ObservationType.continuous, strike_observation=None)
bothfloat_lookback = LookBackPayoff(floating_strike=True, floating_spot=True, 
                                 spot_method=LookbackMethod.geometric_mean, strike_method=LookbackMethod.arithmetic_mean, 
                                 spot_observation=ObservationType.continuous, strike_observation=ObservationType.continuous)
opt_payoff = OptionPayoff(
    option_type=OptionalityType.call,
    exercise=ExerciseType.european)
opt_spec = OptionSpecification(100, OptionTenor(expiry=1))
option = Option(opt_spec, opt_payoff)
S = 100 
r = 0.01
q = 0.1 
t = 10
sigma = 0.2
N = 1000
M = 10000
dt = t/N
Bu = 120 

mcinput = MonteCarloBlackScholesInput(
    S=S, r=r,q=q,t=t,sigma=sigma,
    number_paths=M, 
    number_steps=N,
    discretization=BlackScholesDiscretization.milstein,
    max_workers=1)
bsmc = MonteCarloBlackScholes(inputdata=mcinput)
simulator = bsmc.get(False,False,False)
sim = simulator.sim
test = MonteCarloPricing(sim=sim, option=option, r=r)
payoff_matrix = test.compute_payoff()


def compute_continuation_payoff(discounted_payoff, spot, spot_squared, indexes):
    n = len(indexes)
    ones = np.ones(n)
    Y = discounted_payoff[indexes]
    output = np.zeros(len(discounted_payoff))
    X_list = [ones, spot[indexes], spot_squared[indexes]]
    X = np.transpose(np.reshape(np.concatenate(X_list),(len(X_list),n)))
    coefs = np.linalg.lstsq(X, Y, rcond=None)[0]
    cpayoff = np.transpose(X.dot(coefs))
    output[indexes] = cpayoff
    return output

payoff = payoff_matrix[:,N-1]
dfdt = np.exp(-r*dt)
for i in range(N-2,-1,-1): 
    discounted_payoff = dfdt*payoff
    actual_payoff, spot_vector = payoff_matrix[:,i], sim[:,i]
    spot_squared_vector = spot_vector**2
    indexes = np.where(actual_payoff>0)[0]
    continuation_payoff = compute_continuation_payoff(
        discounted_payoff,spot_vector,
        spot_squared_vector,indexes)
    exercise_indexes = actual_payoff>continuation_payoff
    no_exercise_indexes = np.invert(exercise_indexes)
    payoff[exercise_indexes] = actual_payoff[exercise_indexes]
    payoff[no_exercise_indexes] = discounted_payoff[no_exercise_indexes]


price = dfdt*np.mean(payoff)




class MonteCarloLeastSquareMethod(Enum): 
    linear_spot = 1 
    


@dataclass
class MonteCarloLeastSquare: 
    simulation : np.array 
    volatility_matrix : np.array 
    lookback_matrix : np.array
    payoff_matrix : np.array 
    dt : float 
    r : float 
    option: Option
    option_steps : OptionSteps
    method : MonteCarloLeastSquareMethod
    
    def __post_init__(self): 
        self.spot_squared = self.simulation**2
        self.bermudan_steps = self.option_steps.bermudan

    @staticmethod
    def coefficients(X, Y) -> np.array: 
        return np.linalg.lstsq(X, Y, rcond=None)[0]
    
    def get_X_list(self, i:int, indexes:np.array) -> List[np.array]: 
        n = len(indexes)
        ones = np.ones(n)
        match self.method: 
            case MonteCarloLeastSquareMethod.linear_spot : 
                return [ones, self.spot[indexes, i], 
                        self.spot_squared[indexes,i]]
    
    def compute_continuation_payoff(self, discounted_payoff:np.array, 
                                    i:int, indexes:np.array) -> np.array:
        n = len(indexes)
        Y, output = discounted_payoff[indexes],np.zeros(len(discounted_payoff))
        X = self.get_X_list(i=i, indexes=indexes) 
        cpayoff = np.transpose(X.dot(self.coefficients(X=X, Y=Y)))
        output[indexes] = cpayoff
        return output
    
    def early_exercise_payoff(self, step_range:range): 
        
        payoff, old_i = self.payoff_matrix[:,N-1], N
        #range(N-2,-1,-1)
        for i in step_range: 
            df = np.exp((old_i-i)*self.dt*-self.r)
            old_i = i 
            discounted_payoff = df*payoff
            actual_payoff = self.payoff_matrix[:,i]
            indexes = np.where(actual_payoff>0)[0]
            continuation_payoff = self.compute_continuation_payoff(
                discounted_payoff=discounted_payoff, 
                i=i, indexes=indexes)
            exercise_indexes = actual_payoff>continuation_payoff
            no_exercise_indexes = np.invert(exercise_indexes)
            payoff[exercise_indexes] = actual_payoff[exercise_indexes]
            payoff[no_exercise_indexes] = discounted_payoff[no_exercise_indexes]
        df = np.exp(-self.r*i*self.dt)
        return df*payoff 
    
    def get_range(self): 
        match self.option.payoff.exercise: 
            case ExerciseType.American: 
                pass 
            
    
    def price(self): 
        match  
    
    
    
    
    




    
