from dataclasses import dataclass
import numpy as np 
import matplotlib.pyplot as plt 
from financialmath.instruments.option import *

@dataclass
class FiniteDifferenceGreeks: 
    # deltas
    dS : float 
    dsigma : float 
    dt : float 
    dr : float 
    dq : float 

    #first order 
    V : float 
    V_S_u : float 
    V_sigma_u : float
    V_t_u : float 
    V_r_u : float 
    V_q_u : float   

    #second order 
    V_sigma_d : float = np.nan
    V_sigma_u_S_u : float = np.nan
    V_sigma_u_S_d : float = np.nan
    V_S_d : float = np.nan
    V_t_u_S_u : float = np.nan
    V_t_u_S_d : float = np.nan
    V_t_u_sigma_u : float = np.nan
    V_t_u_sigma_d : float= np.nan
    
    #third order 
    V_S_dd : float = np.nan
    V_S_uu : float = np.nan
    V_sigma_uu : float = np.nan
    V_sigma_dd : float = np.nan

    def price(self): 
        return self.V

    #first order
    def delta(self): 
        return (self.V_S_u-self.V)/self.dS
    
    def vega(self): 
        return (self.V_sigma_u-self.V)/self.dsigma 
    
    def theta(self): 
        return (self.V_t_u-self.V)/self.dt

    def rho(self): 
        return (self.V_r_u-self.V)/self.dt

    def epsilon(self): 
        return (self.V_q_u-self.V)/self.dq
    
    # second order   
    def gamma(self): 
        return (self.V_S_u+self.V_S_d-2*self.V)/(self.S**2) 
    
    def volga(self): 
        return (self.V_sigma_u+self.V_sigma_d-2*self.V)/(self.dsigma**2) 
    
    def vanna(self): 
        delta_up = (self.V_sigma_u_S_u-self.V)/self.dS
        delta_down = (self.V-self.V_sigma_u_S_d)/self.dS
        return (delta_up-delta_down)/self.dsigma
    
    def charm(self): 
        delta_up = (self.V_t_u_S_u-self.V)/self.dS
        delta_down = (self.V-self.V_t_u_S_d)/self.dS
        return (delta_up-delta_down)/self.dt 

    def veta(self): 
        vega_up = (self.V_t_u_sigma_u-self.V)/self.dsigma 
        vega_down = (self.V-self.V_t_u_sigma_d)/self.dsigma 
        return (vega_up-vega_down)/self.dt

    #third order
    def speed(self): 
        delta_up = (self.V_S_u-self.V)/self.dS
        delta_down = (self.V-self.V_S_d)/self.dS
        delta_uu = (self.V_S_uu-self.V_S_u)/self.dS
        delta_dd = (self.V_d - self.V_S_dd)/self.dS
        gamma_up = (delta_uu - delta_up)/self.dS
        gamma_down = (delta_down - delta_dd)/self.dS
        return (gamma_up-gamma_down)/self.dS
    
    def ultima(self): 
        vega_up = (self.V_sigma_u-self.V)/self.dsigma
        vega_down = (self.V-self.V_sigma_d)/self.dsigma
        vega_uu = (self.V_sigma_uu-self.V_sigma_u)/self.dsigma
        vega_dd = (self.V_sigma_d - self.V_sigma_dd)/self.dsigma
        volga_up = (vega_uu - vega_up)/self.dsigma
        volga_down = (vega_down - vega_dd)/self.dsigma
        return (volga_up-volga_down)/self.dsigma

    def color(self): 
        d = (self.dt*(self.dS)**2)
        return (self.V_t_u_S_u+self.V_t_u_S_u-2*self.V_t_u)/d

    def zomma(self): 
        d = (self.dsigma*(self.dS)**2)
        return (self.V_sigma_u_S_u+self.V_sigma_u_S_d-2*self.V_sigma_u)/d
    
@dataclass
class MonteCarloPayoff: 

    strike: float 
    barrier_up: float 
    barrier_down: float 
    gap_trigger: float 
    binary_amount : float 
    rebate: float 

    def __post_init__(self): 
        self.K = self.strike
        self.Bu = self.barrier_up
        self.Bd = self.barrier_down
        self.G = self.gap_trigger
        self.binary_amount = self.binary_amout
        self.rebate = self.specification.rebate

    def barrier_condition(self, spot_vector: np.array) -> np.array: 
        S=spot_vector
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

    def vanilla_payoff(self, spot_vector: np.array) -> np.array: 
        S = spot_vector
        K = self.K
        match self.payoff.option_type: 
            case OptionalityType.call: 
                return np.maximum(S-K,0)
            case OptionalityType.put: 
                return np.maximum(K-S,0)
            case _: 
                return np.repeat(np.nan, self.M) 

    def gap_payoff(self, spot_vector: np.array) -> np.array: 
        S = spot_vector 
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

    def payoff_vector(self, spot_vector:np.array) -> np.array: 
        if self.payoff.gap: 
            payoff = self.gap_payoff(spot_vector=spot_vector)
        else: 
            payoff = self.vanilla_payoff(spot_vector=spot_vector)
        barrier_cond = self.barrier_condition(spot_vector=spot_vector)
        barrier_invcond = np.abs(np.array(barrier_cond) -1)
        payoff = payoff * barrier_cond + barrier_invcond*self.rebate
        if self.payoff.binary: 
            payoff = self.binary_payoff(payoff = payoff)
        return payoff

    def payoff_viewer(self, spot_vector:np.array): 
        payoff = self.payoff_vector(spot_vector=spot_vector)
        plt.plot(self.S, payoff)
        plt.show()

@dataclass
class MonteCarloPricing: 
    option : Option 
    simulation : np.array
    r : float 

    def __post_init__(self): 
        self.t = self.option.specification.tenor.expiry
        self.df = np.exp(-self.r*self.t)
        self.N = self.simulation.shape[1]
        self.option_steps = self.option.specification.get_steps(self.N)



    def strike(self, sim:np.array): 
        pass 

    def breach_barrier(self, sim:np.array)-> np.array: 
        pass







