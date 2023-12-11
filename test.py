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
    option_type=OptionalityType.put,
    exercise=ExerciseType.european)
opt_spec = OptionSpecification(100, OptionTenor(expiry=1))
option = Option(opt_spec, opt_payoff)
S = 100 
r = 0.01
q = 0.1 
t = 1 
sigma = 0.2
N = 100
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













@dataclass
class MonteCarloLeastSquare: 
    simulation : np.array 
    volatility_matrix : np.array 
    lookback_matrix : np.array
    payoff_matrix : np.array 
    dt : float 
    r : float 
    payoff : OptionPayoff
    bermudan_steps : List[int]

    @staticmethod
    def coefficients(X, Y) -> np.array: 
        return np.linalg.lstsq(X, Y, rcond=None)[0]

    def continuation_payoff(self, discounted_payoff:np.array, spot: np.array, 
                           volatility: np.array,pos: np.array, 
                           look_back=None) -> np.array: 
        Y, s, v = discounted_payoff[pos],spot[pos],volatility[pos]
        if not look_back is None: 
            lb = look_back[pos]
            vec_list = [s, s**2,v, v**2, lb, lb**2]
        else: vec_list = [s, s**2,v, v**2]
        XT = np.reshape(np.concatenate(vec_list),(len(vec_list),len(pos)))
        X = np.transpose(XT)
        coefs = self.coefficients(X,Y)
        return X.dot(coefs)
    
    def early_exercise(self, discounted_payoff:np.array, n:int) -> np.array: 
        payoff_vector = self.payoff_matrix[:,n]
        pos = np.where(payoff_vector>0)
        spot, vol = self.simulation[pos,n], self.volatility_matrix[pos,n]
        if self.payoff.is_lookback(): lb = self.lookback_matrix[pos,n]
        else: lb=None
        dpayoff = discounted_payoff[pos]
        cpayoff = self.continuation_payoff(dpayoff,spot,vol)
        



    
