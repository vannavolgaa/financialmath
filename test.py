from financialmath.model.blackscholes.montecarlo import (BlackScholesDiscretization, 
MonteCarloBlackScholes, MonteCarloBlackScholesInput)
from financialmath.model.blackscholes.pde import (PDEBlackScholes, PDEBlackScholesInput)
import numpy as np 
import matplotlib.pyplot as plt 
from financialmath.instruments.option import *
from dataclasses import dataclass
S = 100 
r = 0.01
q = 0.02 
t = 1 
sigma = 0.2
N = 500
M = 10000
dt = t/N

mcinput = MonteCarloBlackScholesInput(
    S=S, r=r,q=q,t=t,sigma=sigma,
    number_paths=M, 
    number_steps=N,
    discretization=BlackScholesDiscretization.milstein,
    first_order_greek=False, 
    second_order_greek=False, 
    third_order_greek=False)
pdeinput = PDEBlackScholesInput(
    S=S, r=r,q=q,t=t,sigma=sigma,
    spot_vector_size=M, 
    number_steps=N)

test = MonteCarloBlackScholes(mcinput).get()
sim = test.sim
#plt.plot(np.transpose(sim))
#plt.show()

opt_payoff = OptionPayoff(OptionalityType.call,ExerciseType.european, forward_start=True)
opt_spec = OptionSpecification(100, OptionTenor(expiry=1, forward_start=0.5))
option = Option(opt_spec, opt_payoff)

@dataclass
class PathLookBack: 
    sim : np.array 
    option : Option 

    def __post_init__(self): 
        self.M, self.N = sim.shape[0], sim.shape[1]
        self.option_steps = self.option.specification.get_steps(self.N)
    
    def fixed_strike(self): 
        K = self.option.specification.strike
        if self.option.payoff.forward_start: 
            return K*self.sim[:,0]
        else: return np.repeat(K,self.M)
    
    def compute_floating(self, sim): 
        n = sim.shape[1]
        match self.option.payoff.lookback_method: 
            case LookbackMethod.geometric_mean: 
                return np.exp(np.sum(np.log(sim),axis=1)/n)
            case LookbackMethod.arithmetic_mean: 
                return np.mean(sim, axis=1)
            case LookbackMethod.maximum: 
                return np.max(sim, axis=1)
            case LookbackMethod.maximum: 
                return np.min(sim, axis=1) 
            case _: return sim[:,self.N-1]
    
    def filter_simulations_for_floating(self): 
        match self.option.payoff.lookback_obervation: 
            case ObservationType.continuous: 
                return self.sim
            case ObservationType.window: 
                s = self.option_steps.lookback_window_begin
                e = self.option_steps.lookback_window_end
                return self.sim[:,s:e]
            case ObservationType.discrete: 
                n_list = self.option_steps.lookback_discrete
                return self.sim[:,n_list]
            case _: return self.sim


@dataclass
class MonteCarloPricing: 
    sim : np.array 
    option : Option 
    r : float

    def __post_init__(self): 
        self.M, self.N = sim.shape[0], sim.shape[1]
        self.option_steps = self.option.specification.get_steps(self.N)
        self.obssim = self.sim[:,self.start_simulation():(self.N-1)]
    
    def start_simulation(self) -> int: 
        if self.option.payoff.forward_start: 
            return self.option_steps.forward_start
        else: return 0
    
    def strike(self) -> int: 
        output = np.zeros((self.M, self.N))
        if self.option.payoff.is_lookback():
            pass 
        else: 
            K = self.option.specification.strike
            if self.option.payoff.forward_start: 
                spot = self.sim[:,self.start_simulation()]
                output[:,:] = K*spot
            else: output[:,:] = K
        return output
    
    def gap(self) -> int: 
        output = np.zeros((self.M, self.N))
        G = self.option.specification.gap_trigger
        if self.option.payoff.forward_start: 
            spot = self.sim[:,self.start_simulation()]
            output[:,:] = G*spot
        else: output[:,:] = G
        return output
    


    def check_barrier(self) -> np.array: 
        ones = np.ones((self.M, self.N))
        match self.option.payoff.barrier_obervation: 
            case ObservationType.continuous: 
                ones = self.option.payofftool(self.obssim).barrier_condition()
            case ObservationType.window: 
                s = self.option_steps.barrier_window_begin
                e = self.option_steps.barrier_window_end
                ptool = self.option.payofftool(self.obssim[:,s:e])
                ones[:,s:e] = ptool.barrier_condition()
            case ObservationType.discrete: 
                n_list = self.option_steps.barrier_discrete
                ptool = self.option.payofftool(self.obssim[:,n_list])
                ones[:,n_list] = ptool.barrier_condition()
            case ObservationType.in_fine: 
                ptool = self.option.payofftool(self.obssim[:,(self.N-1)])
                ones[:,(self.N-1)] = ptool.barrier_condition()
            case _: pass
        return np.cumprod(ones, axis=1)

mcpayoof = MonteCarloPricing(sim, option, r=r)
test2 = MonteCarloPricing(sim, option, r=r).check_barrier()








