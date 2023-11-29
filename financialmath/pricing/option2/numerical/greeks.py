import numpy as np 
from dataclasses import dataclass

@dataclass
class OptionGreeks: 
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
    V_S_u : float = np.nan 
    V_sigma_u : float = np.nan
    V_t_u : float = np.nan
    V_r_u : float = np.nan
    V_q_u : float = np.nan

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


