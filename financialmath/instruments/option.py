from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List
import math
import numpy as np 
from financialmath.instruments.errors import *

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

class LookbackMethod(Enum): 
    geometric_mean = 1 
    minimum = 2 
    maximum = 3
    arithmetic_mean = 4

class ObservationType(Enum): 
    continuous = 1 
    discrete = 2
    window = 3
    in_fine = 4

@dataclass
class LookBackPayoff: 
    floating_strike : bool 
    floating_spot : bool 
    spot_method : LookbackMethod
    strike_method : LookbackMethod
    spot_observation : ObservationType
    strike_observation : ObservationType

@dataclass
class OptionPayoff: 
    option_type : OptionalityType 
    exercise : ExerciseType
    future : bool = False
    binary : bool = False
    gap : bool = False
    forward_start : bool = False
    barrier_type : BarrierType = None
    barrier_observation : ObservationType = None
    lookback : LookBackPayoff = None 
    
    def check_lookback_conditions(self) -> None: 
        p = self.payoff
        if self.is_lookback(): 
            if self.lookback.spot_observation is ObservationType.in_fine: 
                raise LookBackOptionError()
            if self.lookback.strike_observation is ObservationType.in_fine: 
                raise LookBackOptionError()
    
    def get_opposite_barrier(self) -> BarrierType: 
        match self.barrier_type: 
            case BarrierType.up_and_in: 
                return BarrierType.up_and_out
            case BarrierType.up_and_out: 
                return BarrierType.up_and_in
            case BarrierType.down_and_in: 
                return BarrierType.down_and_out
            case BarrierType.down_and_out: 
                return BarrierType.down_and_in
            case BarrierType.double_knock_in: 
                return BarrierType.double_knock_out
            case BarrierType.double_knock_out: 
                return BarrierType.double_knock_in
            case _: return None

    def is_barrier(self) -> bool: 
        if (self.barrier_type is None) or (self.barrier_observation is None):
            return False
        else: return True
    
    def is_lookback(self) -> bool: 
        if lookback is None:return False
        else: return True

    def is_in_barrier(self) -> bool: 
        match self.barrier_type: 
            case BarrierType.up_and_in: return True
            case BarrierType.up_and_out: return False
            case BarrierType.down_and_in: return True
            case BarrierType.down_and_out: return False
            case BarrierType.double_knock_in: return True
            case BarrierType.double_knock_out: return False
            case _: return False
    
    def is_out_barrier(self) -> bool: 
        match self.barrier_type: 
            case BarrierType.up_and_in: return False
            case BarrierType.up_and_out: return True
            case BarrierType.down_and_in: return False
            case BarrierType.down_and_out: return True
            case BarrierType.double_knock_in: return False
            case BarrierType.double_knock_out: return True
            case _: return False

    def is_early_exercise(self) -> bool: 
        match self.exercise:
            case ExerciseType.european: return False
            case ExerciseType.american: return True
            case ExerciseType.bermudan: return True
    
@dataclass
class OptionTenor: 
    expiry : float 
    bermudan : List[float] = field(default_factory=list) 
    barrier_discrete : List[float] = field(default_factory=list) 
    spot_lookback_discrete : List[float] = field(default_factory=list) 
    strike_lookback_discrete : List[float] = field(default_factory=list) 
    barrier_window_begin : float = np.nan 
    spot_lookback_window_begin  : float = np.nan 
    strike_lookback_window_begin  : float = np.nan 
    barrier_window_end  : float = np.nan 
    spot_lookback_window_end  : float = np.nan 
    strike_lookback_window_end  : float = np.nan 
    forward_start : float = np.nan 
    
    def __post__init(self): 
        self.bermudan.sort(), self.strike_lookback_discrete.sort()
        self.spot_lookback_discrete.sort(), self.barrier_discrete.sort()
    
    def check(self, x) -> None: 
        if not np.isnan(x):
            if x<0: raise(NegativeTenorError())
            if x>=self.expiry:raise(TenorConsistencyError())
            if x<=self.forward_start: raise(TenorConsistencyError())
    
    def check_inconsistency_tenors(self) -> None: 
        check_list = [self.expiry, self.bermudan, self.barrier_discrete, 
                      self.spot_lookback_discrete, 
                      self.spot_lookback_window_begin, 
                      self.spot_lookback_window_end, self.barrier_window_begin, 
                      self.barrier_window_end, self.strike_lookback_discrete, 
                      self.strike_lookback_window_begin, 
                      self.strike_lookback_window_end, self.forward_start]
        for cl in check_list: 
            if isinstance(cl, list): [self.check(c) for c in cl]
            else: self.check(cl)
                
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
        self.spot_lookback_discrete = [self.t_to_step(t=t, dt=dt) 
                                for t in tenor.spot_lookback_discrete]
        self.strike_lookback_discrete = [self.t_to_step(t=t, dt=dt) 
                                for t in tenor.strike_lookback_discrete]
        self.spot_lookback_window_begin  = \
            self.t_to_step(t=tenor.spot_lookback_window_begin, dt=dt) 
        self.spot_lookback_window_end  = \
            self.t_to_step(t=tenor.spot_lookback_window_end, dt=dt)
        self.strike_lookback_window_begin  = \
            self.t_to_step(t=tenor.strike_lookback_window_begin, dt=dt) 
        self.strike_lookback_window_end  = \
            self.t_to_step(t=tenor.strike_lookback_window_end, dt=dt)
        self.forward_start = self.t_to_step(t=tenor.forward_start, dt=dt)

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
class Option: 
    specification : OptionSpecification 
    payoff : OptionPayoff
    
    def __post_init__(self): 
        self.payoff.check_lookback_conditions()<
        self.specification.tenor.check_inconsistency_tenors() 
    
@dataclass
class MarketOptionQuotes: 
    bid : float 
    ask : float 
    option : Option 

    def __post_init__(self): 
        self.mid = (self.bid+self.ask)/2


    
