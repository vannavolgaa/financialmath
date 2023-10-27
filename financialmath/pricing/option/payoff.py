from dataclasses import dataclass
import numpy as np 
from financialmath.instruments.option import OptionPayoff
from financialmath.instruments.option import * 
import matplotlib.pyplot as plt 

@dataclass
class PayOffModule: 

    S : float or np.array 
    K : float 
    Bu : float = np.nan 
    Bd : float = np.nan 
    G : float = np.nan 
    binary_amount : float = np.nan 
    rebate : float = 0 

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
                condition = (S<b_down) & (S>b_up)
            case BarrierType.double_knock_out:
                condition = (S>b_down) & (S<b_up) 
            case _: 
                condition = np.repeat(1, M)
        return condition.astype(int) 

    def vanilla_payoff(self, option_type:OptionalityType): 
        S = self.S 
        K = self.K
        match option_type: 
            case OptionalityType.call: 
                return np.maximum(S-K,0)
            case OptionalityType.put: 
                return np.maximum(K-S,0)
            case _: 
                return np.repeat(np.nan, self.M) 

    def gap_payoff(self, option_type:OptionalityType): 
        S = self.S 
        G = self.G
        K = self.K
        match option_type: 
            case OptionalityType.call: 
                return (S>G).astype(int)*(S-K)
            case OptionalityType.put: 
                return (S<G).astype(int)*(K-S)
            case _: 
                return np.repeat(np.nan, self.M)
    
    def binary_payoff(self, payoff: np.array):
        return (abs(payoff)>0).astype(int)*self.binary_amount*np.sign(payoff)

    def payoff(self, option_payoff:OptionPayoff): 
        if option_payoff.gap: payoff = self.gap_payoff(
            option_type = option_payoff.option_type)
        else: payoff = self.vanilla_payoff(
            option_type = option_payoff.option_type)
        barrier_condition = self.barrier_condition(
            barrier_type=option_payoff.barrier_type)
        inverse_barrier_condition = np.abs(np.array(barrier_condition) -1)
        payoff = payoff * barrier_condition + inverse_barrier_condition*self.rebate
        if option_payoff.binary: 
            payoff = self.binary_payoff(payoff = payoff)
        return payoff

    def payoff_viewer(self, option_payoff:OptionPayoff): 
        payoff = self.payoff(option_payoff=option_payoff)
        plt.plot(self.S, payoff)
        plt.show()


