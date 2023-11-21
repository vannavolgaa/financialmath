from dataclasses import dataclass
from enum import Enum
import numpy as np
from financialmath.tools.simulation import RandomGenerator, RandomGeneratorType, NormalDistribution

#TODO: add thread option in simulation

class BlackScholesDiscretization(Enum): 
    euler = 1 
    milstein = 2

@dataclass 
class MonteCarloBlackScholesInput: 
    S : float 
    r : float 
    q : float 
    sigma : float 
    t : float 
    number_steps : int = 400
    number_paths : int = 10000
    future : bool = False
    greeks : bool = True
    randoms_generator : RandomGeneratorType = RandomGeneratorType.antithetic
    discretization: BlackScholesDiscretization=BlackScholesDiscretization.euler
    ds : float = 0.01 
    dv : float = 0.01 
    dr : float = 0.01 
    dq : float = 0.01 

@dataclass
class MonteCarloBlackScholesOutput: 
    initial : np.array 
    spot_squareup : np.array = None
    spot_up : np.array = None
    spot_down : np.array = None
    spot_squaredown : np.array = None
    vol_up : np.array = None
    vol_down : np.array = None
    r_up : np.array = None
    q_up : np.array = None 
    ds : float = 0 
    dv : float = 0 
    dr : float = 0 
    dq : float = 0 

@dataclass
class MonteCarloBlackScholes: 

    inputdata:  MonteCarloBlackScholesInput

    def __post_init__(self): 
        self.N = self.inputdata.number_steps
        self.M = self.inputdata.number_paths
        self.dt = self.inputdata.t/self.N
        self.Z = self.generate_randoms()
        self.sigma = self.inputdata.sigma 
        self.r = self.inputdata.r 
        self.q = self.inputdata.q 
        self.S = self.inputdata.S 
        self.ds = self.inputdata.ds
        self.dv = self.inputdata.dv
        self.dr = self.inputdata.dr
        self.dq = self.inputdata.dq

    def generate_randoms(self) -> np.array:
        generator =  RandomGenerator(
            probability_distribution=NormalDistribution(), 
            generator_type=self.inputdata.randoms_generator)
        randoms = generator.generate(N=self.M*self.N)
        return np.reshape(randoms, (self.M,self.N))

    def drift(self, sigma:float, r:float=0, q:float=0) -> float: 
        if self.inputdata.future: 
             return -.5*(sigma**2)*self.dt
        else: 
            return ((r-q)-.5*(sigma**2))*self.dt
    
    def diffusion(self, sigma:float) -> np.array: 
        return sigma * np.sqrt(self.dt)*self.Z 
    
    def milstein_correction(self, sigma:float) -> np.array: 
        return -.5*(sigma**2)*self.dt*(self.Z**2 - 1) 
    
    def euler_moneyness(self, sigma:float, r:float=0, q:float=0) -> np.array:
        drift = self.drift(sigma=sigma, r=r, q=q) 
        diffusion = self.diffusion(sigma=sigma) 
        return np.cumprod(np.exp(drift+diffusion), axis = 1)

    def simulation_no_greek(self) -> MonteCarloBlackScholesOutput: 
        euler_moneyness= self.euler_moneyness(sigma=self.sigma, r=self.r, q=self.q)
        if self.inputdata.discretization is BlackScholesDiscretization.milstein:
            return MonteCarloBlackScholesOutput(
                initial=self.milstein_price(self.S,self.sigma, euler_moneyness)) 
        else: 
            return MonteCarloBlackScholesOutput(
                initial=self.euler_price(self.S, euler_moneyness))   
    
    def simulation_euler_with_greeks(self)-> MonteCarloBlackScholesOutput: 

        q, sigma, r, dv, dr, dq, ds = self.q, self.sigma, self.r, \
                                    self.dv,self.dr,self.dq, self.ds
        S = self.S
        euler_moneyness= self.euler_moneyness(sigma,r,q)
        euler_moneyness_vol_up= self.euler_moneyness(sigma+dv,r,q)
        euler_moneyness_vol_down = self.euler_moneyness(sigma-dv,r,q)
        euler_moneyness_r_up= self.euler_moneyness(sigma,r+dr,q)
        euler_moneyness_q_up= self.euler_moneyness(sigma,r,q+dq) 
        return MonteCarloBlackScholesOutput(
            initial = S * euler_moneyness, 
            spot_squareup = (S+2*ds)* euler_moneyness, 
            spot_up = (S+ds)* euler_moneyness, 
            spot_down = (S-ds)* euler_moneyness, 
            spot_squaredown = (S-2*ds)* euler_moneyness, 
            vol_up = S* euler_moneyness_vol_up, 
            vol_down = S* euler_moneyness_vol_down, 
            r_up = S*euler_moneyness_r_up, 
            q_up= S*euler_moneyness_q_up, 
            ds = ds, dv=dv, dr=dr, dq=dq
        )

    def simulation_milstein_with_greeks(self)-> MonteCarloBlackScholesOutput: 
        sigma = self.sigma
        milstein_correction = self.milstein_correction(sigma)
        milstein_correction_vol_up = self.milstein_correction(sigma+self.dv)
        milstein_correction_vol_down = self.milstein_correction(sigma-self.dv) 
        output = self.simulation_euler_with_greeks()
        output.initial = output.initial+milstein_correction
        output.spot_squareup = output.spot_squareup+milstein_correction
        output.spot_up = output.spot_up+milstein_correction
        output.spot_squaredown = output.spot_squaredown+milstein_correction
        output.spot_down = output.spot_down+milstein_correction
        output.vol_up = output.vol_up+milstein_correction_vol_up
        output.vol_down = output.vol_down+milstein_correction_vol_down
        output.r_up = output.r_up+milstein_correction
        output.q_up = output.q_up+milstein_correction
        return output 
    
    def simulation(self) -> MonteCarloBlackScholesOutput: 
        discretization = self.inputdata.discretization
        if self.inputdata.greeks: 
            if discretization is BlackScholesDiscretization.milstein: 
                return self.simulation_milstein_with_greeks()
            else: 
                return self.simulation_euler_with_greeks()
        else: return self.simulation_no_greek()
        
    



    