from financialmath.model.blackscholes.montecarlo import (BlackScholesDiscretization, 
MonteCarloBlackScholes, MonteCarloBlackScholesInput)
from financialmath.model.blackscholes.pde import (PDEBlackScholes, PDEBlackScholesInput)
import numpy as np 
import matplotlib.pyplot as plt 
from financialmath.instruments.option import *
from dataclasses import dataclass
from financialmath.pricing.option.pde import PDEBlackScholesValuation
import matplotlib.pyplot as plt

@dataclass
class MonteCarloLookback: 
    sim : np.array 
    option_steps : OptionSteps
    forward_start : bool 
    lookback_method : LookbackMethod
    observation_type : ObservationType

    def __post_init__(self): 
        self.N = self.option_steps.N
        self.M = self.sim.shape[0]
        self.fstart_step = self.option_steps.forward_start
    
    @staticmethod
    def progressive_npfun(vectors:np.array, fun:object) -> np.array: 
        emptyvec = np.zeros(vectors.shape)
        emptyvec[:,0] = vectors[:,0]
        for i in range(1,emptyvec.shape[1]): 
            emptyvec[:,i] = fun(vectors[:,0:i], axis=1)
        return emptyvec

    def compute_lookback_method(self, vectors:np.array) -> np.array: 
        match self.lookback_method: 
            case LookbackMethod.geometric_mean: 
                return np.log(self.progressive_npfun(np.exp(vectors),np.mean))
            case LookbackMethod.arithmetic_mean:
                return self.progressive_npfun(vectors,np.mean) 
            case LookbackMethod.minimum:
                return self.progressive_npfun(vectors,np.min) 
            case LookbackMethod.maximum: 
                return self.progressive_npfun(vectors,np.max) 
    
    def continuous_observation(self)-> np.array:
        if self.forward_start:sim = self.sim[:,self.fstart_step:self.N] 
        else: sim = self.sim
        return self.compute_lookback_method(sim)
    
    @staticmethod
    def repeat_vector_into_matrix(vec:np.array, M:int, N:int) -> np.array: 
        return np.reshape(np.repeat(vec, N), (M,N))
    
    def window_observation(self) -> np.array: 
        s = self.option_steps.lookback_window_begin
        e = self.option_steps.lookback_window_end
        if self.forward_start: 
            n = self.N - self.fstart_step 
            out = np.zeros((self.sim.shape[0], n))
            out[:,0:s] = self.sim[:,self.fstart_step:s]
        else: 
            out = np.zeros(self.sim.shape)
            out[:,0:s] = self.sim[:,0:s]
        new_N = out.shape[1]
        out[:,s:e] = self.compute_lookback_method(self.sim[:,s:e])
        n = new_N - e
        final_vector = self.repeat_vector_into_matrix(out[:,e-1],self.M,n)
        out[:,e:new_N] = final_vector
        return out
    
    def discrete_observation(self) -> np.array: 
        pass


opt_payoff = OptionPayoff(
    option_type=OptionalityType.call,
    exercise=ExerciseType.european, 
    barrier_type=BarrierType.down_and_out, 
    barrier_observation=ObservationType.continuous, forward_start=True)
opt_spec = OptionSpecification(100, OptionTenor(expiry=1, forward_start=0.5), barrier_up=120, barrier_down=80, rebate=0)
option = Option(opt_spec, opt_payoff)


S = 100 
r = 0.01
q = 0.1 
t = 1 
sigma = 0.2
N = 500
M = 3
dt = t/N
Bu = 120 


option_steps = option.specification.get_steps(N=N)

option_steps.forward_start

mcinput = MonteCarloBlackScholesInput(
    S=S, r=r,q=q,t=t,sigma=sigma,
    number_paths=M, 
    number_steps=N,
    discretization=BlackScholesDiscretization.milstein,
    max_workers=1)

bsmc = MonteCarloBlackScholes(inputdata=mcinput)
test = bsmc.get(False,False,False)
sim = test.sim


steps = OptionSteps(
    OptionTenor(
    expiry=1, forward_start=0.3, 
    lookback_window_begin=0.5, lookback_window_end=0.75), 
    N = N)


mclb = MonteCarloLookback(
    sim = sim, 
    option_steps=steps, 
    forward_start=False, 
    lookback_method=LookbackMethod.geometric_mean, 
    observation_type=ObservationType.continuous
)

test = mclb.window_observation()

plt.plot(np.transpose(test))
#plt.plot(np.transpose(sim[:,steps.forward_start:steps.N]))
plt.plot(np.transpose(sim))
plt.show()