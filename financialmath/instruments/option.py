from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List
import math

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
        self.steps = OptionSteps(tenor=self.tenor, N=N)

@dataclass
class Option: 
    specification : OptionSpecification 
    payoff : OptionPayoff 
    chooser : bool = False 


