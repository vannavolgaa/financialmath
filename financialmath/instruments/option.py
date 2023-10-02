from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List
import math

class OptionalityType(Enum): 
    call = 1 
    put = 2 
    chooser = 3 

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

class LookbackStrike(Enum): 
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
    option : OptionalityType 
    exercise : ExerciseType
    binary : Optional[bool] = False
    forward_start : Optional[bool] = False
    barrier_type : Optional[BarrierType] = None
    barrier_obervation_type : Optional[ObservationType] = None
    lookback_strike : Optional[LookbackStrike] = None  
    lookback_method : Optional[LookbackMethod] = None    
    lookback_obervation : Optional[ObservationType] = None   

class OptionPayoffList(Enum): 
    unknown = None 
    european_vanilla_call = OptionPayoff(option=OptionalityType.call,
                            exercise=ExerciseType.european) 
    european_vanilla_put = OptionPayoff( option=OptionalityType.put, 
                            exercise=ExerciseType.european)                       
    european_binary_call = OptionPayoff(option=OptionalityType.call, 
                            exercise=ExerciseType.european, binary = True)
    european_binary_put = OptionPayoff(option=OptionalityType.put, 
                            exercise=ExerciseType.european, binary=True)
    american_vanilla_call = OptionPayoff(option=OptionalityType.call, 
                            exercise=ExerciseType.american) 
    american_vanilla_put = OptionPayoff(option=OptionalityType.put, 
                            exercise=ExerciseType.american) 
    bermudan_vanilla_call = OptionPayoff(option=OptionalityType.call, 
                            exercise=ExerciseType.bermudan) 
    bermudan_vanilla_put = OptionPayoff(option=OptionalityType.put, 
                            exercise=ExerciseType.bermudan) 

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
    choser: Optional[float] = math.nan 
    forward : Optional[float] = math.nan 

@dataclass
class OptionSpecification: 
    strike : float 
    tenor : OptionTenor
    forward: bool = False
    barrier_up : float = math.nan
    barrier_down : float = math.nan

@dataclass
class Option: 
    specification : OptionSpecification
    payoff : OptionPayoff

    def payoff_type(self): 
        payoff_list = list(OptionPayoffList)
        try: return [p for p in payoff_list if p.value == self.payoff][0]
        except: return OptionPayoffList.unknown 
    
class CreateOption: 

    @staticmethod
    def european_vanilla_call(strike: float, expiry: float, fwd = False) -> Option:
        tenor = OptionTenor(expiry=expiry)
        spec = OptionSpecification(strike=strike, tenor=tenor,forward=fwd)
        return Option(specification=spec, 
                payoff=OptionPayoffList.european_vanilla_call.value)

    @staticmethod
    def european_vanilla_put(strike: float, expiry: float, fwd = False) -> Option:
        tenor = OptionTenor(expiry=expiry)
        spec = OptionSpecification(strike=strike, tenor=tenor, forward=fwd)
        return Option(specification=spec, payoff=OptionPayoffList.european_vanilla_put.value)
    
