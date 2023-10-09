from dataclasses import dataclass
import numpy as np 
from financialmath.instruments.option import Option
from financialmath.instruments.option import * 

@dataclass
class PayOffModule: 

    S : float or np.array 
    K : float 
    Bu : float = np.nan 
    Bd : float = np.nan 
    G : float = np.nan 
    binary_payoff : float = np.nan 

    def barrier_condition(self, barrier_type: BarrierType): 
        S=self.S
        b_up = self.Bu
        b_down = self.Bd
        M = len(S)
        match barrier_type: 
            case BarrierType.up_and_in:
                condition = (S>b_up)
            case BarrierType.up_and_out:
                condition = (S<b_up)
            case BarrierType.down_and_in: 
                condition = (S<b_down)
            case BarrierType.down_and_out: 
                condition = (S>b_down)
            case BarrierType.double_knock_in:
                condition = (S<b_down) and (S>b_up)
            case BarrierType.double_knock_in:
                condition = (S>b_down) and (S<b_up) 
            case _: 
                condition = np.repeat(1, M)
        return condition.astype(int) 

    def vanilla_condition(self): 
        S = self.S 
        match self.option.payoff.option: 
            case OptionalityType.call: 
                return (S>K).astype(int)
            case OptionalityType.put: 
                return (S<K).astype(int)
            case _: 
                return np.repeat(np.nan, self.M) 

    def gap_condition(self): 
        S = self.S 
        G = self.G
        match self.option.payoff.option: 
            case OptionalityType.call: 
                return (S>G).astype(int)
            case OptionalityType.put: 
                return (S<G).astype(int)
            case _: 
                return np.repeat(np.nan, self.M)
    
    def binary_condition(self, payoff: np.array):
        return (payoff>0).astype(int) 

    def payoff(self): 
        pass 

