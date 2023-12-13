from typing import NamedTuple
import numpy as np
from dataclasses import dataclass
from financialmath.instruments.option import (
    Option, 
    BarrierType,  
    OptionalityType, 
    OptionPayoff)

@dataclass
class OptionPayOffTool: 
    spot:np.array or float
    strike: np.array or float
    barrier_up: np.array or float
    barrier_down: np.array or float
    gap_trigger: np.array or float
    binary_amount: np.array or float
    rebate: np.array or float
    payoff : OptionPayoff

    def barrier_condition(self) -> np.array or bool: 
        S, b_up, b_down=self.spot, self.barrier_up, self.barrier_down
        match self.payoff.barrier_type: 
            case BarrierType.up_and_in: condition = (S>b_up)
            case BarrierType.up_and_out: condition = (S<b_up)
            case BarrierType.down_and_in: condition = (S<b_down)
            case BarrierType.down_and_out: condition = (S>b_down)
            case BarrierType.double_knock_in: condition = (S<b_down)&(S>b_up)
            case BarrierType.double_knock_out: condition = (S>b_down)&(S<b_up)
        return condition.astype(int) 
    
    def rebate_payoff(self) -> np.array or float: 
        barrier_cond = self.barrier_condition()
        barrier_invcond = np.abs(np.array(barrier_cond) -1)
        return barrier_invcond*self.rebate

    def vanilla_payoff(self) -> np.array or float: 
        S,K = self.spot, self.strike
        match self.payoff.option_type: 
            case OptionalityType.call: return np.maximum(S-K,0)
            case OptionalityType.put: return np.maximum(K-S,0)

    def gap_payoff(self) -> np.array or float: 
        S, G, K = self.spot,self.gap_trigger, self.strike
        match self.payoff.option_type: 
            case OptionalityType.call: return (S>G).astype(int)*(S-K)
            case OptionalityType.put: return (S<G).astype(int)*(K-S)
    
    def binary_payoff(self, payoff: np.array) -> np.array or float:
        return (abs(payoff)>0).astype(int)*self.binary_amount*np.sign(payoff)
    
    def payoff_vector(self) -> np.array or float: 
        if self.payoff.gap: payoff = self.gap_payoff()
        else: payoff = self.vanilla_payoff()
        if self.payoff.is_barrier(): 
            barrier_cond = self.barrier_condition()
            rebate_payoff = self.rebate_payoff()
            payoff = payoff * barrier_cond + rebate_payoff
        if self.payoff.binary: payoff = self.binary_payoff(payoff = payoff)
        return payoff
    
    def payoff_vector_no_barrier(self) -> np.array or float: 
        if self.payoff.gap: payoff = self.gap_payoff()
        else: payoff = self.vanilla_payoff()
        if self.payoff.binary: payoff = self.binary_payoff(payoff = payoff)
        return payoff
     
class OptionGreeks(NamedTuple): 
    delta: float = np.nan
    vega: float = np.nan
    theta: float = np.nan
    rho: float = np.nan
    epsilon: float = np.nan
    gamma: float = np.nan
    vanna: float = np.nan
    volga: float = np.nan
    charm: float = np.nan
    veta: float = np.nan
    vera: float = np.nan
    speed: float = np.nan
    zomma: float = np.nan
    color: float = np.nan
    ultima: float = np.nan

class OptionValuationResult(NamedTuple): 
    instrument : Option
    inputdata : classmethod
    price : float 
    sensitivities : OptionGreeks
    method : str
    time_taken : float 
    