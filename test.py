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
    option_type=OptionalityType.call,
    exercise=ExerciseType.european)
opt_spec = OptionSpecification(100, OptionTenor(expiry=1))
option = Option(opt_spec, opt_payoff)
S = 100 
r = 0.01
q = 0.1 
t = 1 
sigma = 0.2
N = 100
M = 50000
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
payoffs = test.compute_payoff()

pos = np.where(payoffs[:,N-2]>0)
actual_payoff = payoffs[pos,N-2]
Y = np.exp(-r*dt)*payoffs[pos[0],N-1]
n_1_spot = sim[pos[0],N-2]
n_1_spot_square =n_1_spot**2
X = np.transpose(np.reshape(np.concatenate([n_1_spot, n_1_spot_square]),(2,len(pos[0]))))

#xxt_inv = np.linalg.inv(np.dot(X, np.transpose(X)))

test3 = np.linalg.lstsq(X, Y, rcond=None)[0]

expected_payoff = np.transpose(X.dot(test3))

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
        



    
