from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List
import math
import numpy as np 
import matplotlib.pyplot as plt 

class OptionalityType(Enum): 
    call = 1 
    put = 2 
  
class UnderlyingType(Enum): 
    spot = 1
    forward = 2

class ExerciseType(Enum): 
    european = 1 
    american = 2 
    bermudan = 3

class BarrierType(Enum): 
    up_and_in = 1 
    up_and_out = 2
    down_and_in = 3 
    down_and_out = 4
    double_knock_in = 5
    double_knock_out = 6

class LookbackStrikeType(Enum): 
    floating_strike = 1 
    fixed_strike = 2

class LookbackMethod(Enum): 
    geometric_mean = 1 
    minimum = 2 
    maximum = 3
    arithmetic_mean = 4

class ObservationType(Enum): 
    continuous = 1 
    discrete = 2
    window = 3

@dataclass
class OptionPayoff: 
    option_type : OptionalityType 
    exercise : ExerciseType
    future : bool = False
    binary : Optional[bool] = False
    gap : Optional[bool] = False
    forward_start : bool = False
    barrier_type : Optional[BarrierType] = None
    barrier_obervation_type : Optional[ObservationType] = None
    lookback_strike : Optional[LookbackStrikeType] = None  
    lookback_method : Optional[LookbackMethod] = None    
    lookback_obervation : Optional[ObservationType] = None   
    
@dataclass
class OptionTenor: 
    expiry : float 
    bermudan : List[float] = field(default_factory=list) 
    barrier_discrete : List[float] = field(default_factory=list) 
    lookback_discrete : List[float] = field(default_factory=list) 
    barrier_window_begin : Optional[float] = math.nan 
    lookback_window_begin  : Optional[float] = math.nan 
    barrier_window_end  : Optional[float] = math.nan 
    lookback_window_end  : Optional[float] = math.nan 
    forward_start : Optional[float] = math.nan 

@dataclass
class OptionSteps:
    tenor : OptionTenor 
    N : int 

    @staticmethod
    def t_to_step(t, dt): 
        factor = t/dt
        if not math.isnan(factor):
            return round(t/dt)
        else: return math.nan

    def __post_init__(self): 
        tenor = self.tenor
        N = self.N 
        dt = tenor.expiry/N 
        self.expiry = N
        self.bermudan = [self.t_to_step(t=t, dt=dt) for t in tenor.bermudan]
        self.barrier_discrete = [self.t_to_step(t=t, dt=dt) 
                                for t in tenor.barrier_discrete]
        self.barrier_window_end  = self.t_to_step(t=tenor.barrier_window_end, dt=dt)
        self.barrier_window_begin = self.t_to_step(t=tenor.barrier_window_begin, dt=dt) 
        self.lookback_discrete = [self.t_to_step(t=t, dt=dt) 
                                for t in tenor.lookback_discrete]
        self.lookback_window_begin  = self.t_to_step(t=tenor.lookback_window_begin, dt=dt) 
        self.lookback_window_end  = self.t_to_step(t=tenor.lookback_window_end, dt=dt)
        #self.choser = self.t_to_step(t=tenor.choser, dt=dt)

@dataclass
class OptionSpecification: 
    strike : float 
    tenor : OptionTenor
    rebate : float = 0
    barrier_up : float = math.nan
    barrier_down : float = math.nan
    gap_trigger : float = math.nan 
    binary_amout : float = math.nan 

    def get_steps(self, N:int) -> OptionSteps:
        return OptionSteps(tenor=self.tenor, N=N)

@dataclass
class PayOffObject: 

    specification : OptionSpecification
    payoff : OptionPayoff
    S : np.array

    def __post_init__(self): 
        self.K = self.specification.strike
        self.Bu = self.specification.barrier_up
        self.Bd = self.specification.barrier_down
        self.R = self.specification.rebate
        self.G = self.specification.gap_trigger
        self.binary_amount = self.specification.binary_amout

    def barrier_condition(self) -> np.array: 
        S=self.S
        b_up = self.Bu
        b_down = self.Bd
        match self.payoff.barrier_type: 
            case BarrierType.up_and_in:
                condition = (S>b_up)
            case BarrierType.up_and_out:
                condition = (S<b_up)
            case BarrierType.down_and_in: 
                condition = (S<b_down)
            case BarrierType.down_and_out: 
                condition = (S>b_down)
            case BarrierType.double_knock_in:
                condition = (S<b_down) & (S>b_up)
            case BarrierType.double_knock_out:
                condition = (S>b_down) & (S<b_up) 
            case _: 
                condition = np.repeat(1, len(S))
        return condition.astype(int) 

    def vanilla_payoff(self) -> np.array: 
        S = self.S 
        K = self.K
        match self.payoff.option_type: 
            case OptionalityType.call: 
                return np.maximum(S-K,0)
            case OptionalityType.put: 
                return np.maximum(K-S,0)
            case _: 
                return np.repeat(np.nan, self.M) 

    def gap_payoff(self) -> np.array: 
        S = self.S 
        G = self.G
        K = self.K
        match self.payoff.option_type: 
            case OptionalityType.call: 
                return (S>G).astype(int)*(S-K)
            case OptionalityType.put: 
                return (S<G).astype(int)*(K-S)
            case _: 
                return np.repeat(np.nan, self.M)
    
    def binary_payoff(self, payoff: np.array) -> np.array:
        return (abs(payoff)>0).astype(int)*self.binary_amount*np.sign(payoff)

    def payoff_vector(self) -> np.array: 
        opttype = self.payoff.option_type
        barrtype = self.payoff.barrier_type
        if self.payoff.gap: 
            payoff = self.gap_payoff(option_type = opttype)
        else: 
            payoff = self.vanilla_payoff(option_type = opttype)
        barrier_cond = self.barrier_condition(barrier_type=barrtype)
        barrier_invcond = np.abs(np.array(barrier_cond) -1)
        payoff = payoff * barrier_cond + barrier_invcond*self.rebate
        if self.payoff.binary: 
            payoff = self.binary_payoff(payoff = payoff)
        return payoff

    def payoff_viewer(self): 
        payoff = self.payoff_vector()
        plt.plot(self.S, payoff)
        plt.show()

@dataclass
class Option: 
    specification : OptionSpecification 
    payoff : OptionPayoff 

    def payoff_object(self, S: np.array): 
        return PayOffObject(self.specification, self.payoff, S)


