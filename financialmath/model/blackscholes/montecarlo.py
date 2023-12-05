from dataclasses import dataclass
from enum import Enum
from typing import List
import numpy as np
import time
from financialmath.tools.simulation import RandomGenerator, RandomGeneratorType, NormalDistribution
from financialmath.tools.tool import MainTool


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
    randoms_generator : RandomGeneratorType = RandomGeneratorType.antithetic
    discretization: BlackScholesDiscretization=BlackScholesDiscretization.euler
    dS : float = 0.01 
    dsigma : float = 0.01 
    dr : float = 0.01 
    dq : float = 0.01 
    max_workers : bool = 4

@dataclass
class EulerBlackScholesSimulation: 
    S: float 
    r: float 
    q: float 
    sigma: float 
    dt: float 
    Z: float 
    future : bool = False

    def drift(self) -> float: 
        if self.future: return -.5*self.dt*(self.sigma**2)
        else: return (self.r-self.q-.5*(self.sigma**2))*self.dt

    def diffusion(self) -> np.array: 
       return self.sigma*np.sqrt(self.dt)*self.Z

    def moneyness_simulation(self)-> np.array: 
        return np.cumprod(np.exp(self.drift()+self.diffusion()), axis=1)
    
    def spot_simulation(self) -> np.array: 
        return self.S*self.moneyness_simulation()

@dataclass
class MilsteinBlackScholesSimulation: 
    S: float 
    r: float 
    q: float 
    sigma: float 
    dt : float 
    Z : float 
    future : bool = True

    def correction(self) -> np.array: 
        return -.5*self.dt*(self.Z**2 - 1)*(self.sigma**2)
    
    def spot_simulation(self) -> np.array: 
        euler = EulerBlackScholesSimulation(
            self.S,self.r, self.q, self.sigma, self.dt, self.Z, self.future)
        return euler.spot_simulation() + self.correction()
      
@dataclass
class MonteCarloBlackScholesOutput: 

    sim : np.array 
    t_vector : np.array
    steps_vector : np.array
    time_taken : float 

    #first order paths
    sim_S_up : np.array = None
    sim_sigma_up : np.array = None
    sim_t_up : np.array = None
    sim_r_up: np.array = None
    sim_q_up : np.array = None
    #second order paths
    sim_sigma_down : np.array = None
    sim_sigma_up_S_up : np.array = None
    sim_sigma_up_S_down : np.array = None
    sim_S_down : np.array = None
    sim_t_up_S_up : np.array = None
    sim_t_up_S_down : np.array = None
    sim_t_up_sigma_up : np.array = None
    sim_t_up_sigma_down : np.array = None
    #third order paths
    sim_S_uu : np.array = None
    sim_S_dd : np.array = None
    sim_sigma_dd : np.array = None
    sim_sigma_uu : np.array = None
    
    dS : float = 0.01
    dsigma : float = 0.01
    dr : float = 0.01 
    dq : float = 0.01
    dt : float = 0.01

class MonteCarloBlackScholes: 

    def __init__(self, inputdata: MonteCarloBlackScholesInput): 
        self.start = time.time()
        self.inputdata = inputdata
        self.t, self.S, self.sigma, self.r, self.q = inputdata.t, inputdata.S,\
            inputdata.sigma, inputdata.r, inputdata.q
        self.dS, self.ds, self.dr, self.dq = inputdata.dS, inputdata.dsigma,\
            inputdata.dr, inputdata.dq
        self.N, self.M = inputdata.number_steps, inputdata.number_paths
        self.dt = self.t/self.N
        self.Z = self.generate_randoms()

    def t_vector(self) -> np.array: 
        return np.cumsum(np.repeat(self.dt, self.N))
    
    def step_vector(self) -> np.array: 
        return np.cumsum(np.repeat(1, self.N))

    def generate_randoms(self) -> np.array:
        N,M = self.N, self.M
        generator =  RandomGenerator(
            probability_distribution=NormalDistribution(), 
            generator_type=self.inputdata.randoms_generator)
        randoms = generator.generate(N=M*N)
        return np.reshape(randoms, (M,N))

    def args_simulation_list(self, greek1: bool=True, greek2:bool=True, 
                        greek3:bool=True) -> List[tuple]: 
        s, ds, dt, r, q, dr, dq, S, dS= self.sigma, self.ds,\
        self.dt, self.r, self.q,self.dr, self.dq, self.S, self.dS
        t = self.t
        N=self.inputdata.number_steps
        dt_up = (t+dt)/N
        args_list = [(S, r, q, s, dt, 0)]
        if greek1: 
            new_args = [(S+dS, r, q, s, dt, 1), 
                        (S, r+dr, q, s, dt, 2),
                        (S, r, q+dq, s, dt, 3), 
                        (S, r, q, s+ds, dt, 4), 
                        (S, r, q, s, dt_up, 5)] 
            args_list = args_list+new_args
            if greek2: 
                new_args = [(S-dS, r, q, s, dt, 6), 
                            (S, r, q, s-ds, dt, 7),
                            (S+dS, r, q, s+ds, dt, 8), 
                            (S-dS, r, q, s+ds, dt, 9), 
                            (S+dS, r, q, s, dt_up, 10),
                            (S-dS, r, q, s, dt_up, 11),
                            (S, r, q, s+ds, dt_up, 12),
                            (S, r, q, s-ds, dt_up, 13)] 
                args_list = args_list+new_args 
                if greek3: 
                    new_args = [(S+2*dS, r, q, s, dt, 14), 
                                (S-2*dS, r, q, s, dt, 15),
                                (S, r, q, s+2*ds, dt, 16), 
                                (S, r, q, s-2*ds, dt, 17)] 
                    args_list = args_list+new_args
        return args_list

    def compute_simulation(self, arg:tuple[float]) -> np.array: 
        S,r,q,sigma,dt,id = arg[0],arg[1],arg[2],arg[3],arg[4],arg[5]
        Z, d = self.Z, self.inputdata.discretization
        if d is BlackScholesDiscretization.milstein: 
            simulator = MilsteinBlackScholesSimulation(
                S=S, r=r, q=q, sigma=sigma,Z=Z,
                dt=dt, future=self.inputdata.future)
        else: 
            simulator = EulerBlackScholesSimulation(
                S=S, r=r, q=q, sigma=sigma,Z=Z,
                dt=dt,future=self.inputdata.future)
        return {id:simulator.spot_simulation()}

    def get_simulations(self, greek1: bool=True, greek2:bool=True, 
                        greek3:bool=True) -> dict[int, np.array]: 
        args = self.args_simulation_list(greek1, greek2, greek3)
        simulations = MainTool.send_task_with_futures(self.compute_simulation,
                    args, max_workers=self.inputdata.max_workers)
        return MainTool.listdict_to_dictlist(simulations)

    def get(self, first_order_greek:bool = True, 
            second_order_greek:bool = True, 
            third_order_greek:bool = True) -> MonteCarloBlackScholesOutput: 
        simulations = self.get_simulations(first_order_greek,
                                           second_order_greek,
                                           third_order_greek)
        end = time.time()
        output = MonteCarloBlackScholesOutput(
            sim = simulations[0], 
            t_vector=self.t_vector(), 
            steps_vector=self.step_vector(),
            time_taken=end-self.start, 
            dS = self.dS, dsigma = self.ds, 
            dt = self.dt, dr = self.dr, dq=self.dq)
        if first_order_greek: 
            output.sim_S_up = simulations[1] 
            output.sim_sigma_up = simulations[4] 
            output.sim_t_up = simulations[5] 
            output.sim_r_up = simulations[2] 
            output.sim_q_up = simulations[3] 
            if second_order_greek: 
                output.sim_sigma_down = simulations[7] 
                output.sim_sigma_up_S_up = simulations[8] 
                output.sim_sigma_up_S_down = simulations[9] 
                output.sim_S_down = simulations[6] 
                output.sim_t_up_S_up = simulations[10] 
                output.sim_t_up_S_down = simulations[11] 
                output.sim_t_up_sigma_up = simulations[12] 
                output.sim_t_up_sigma_down = simulations[13] 
                if third_order_greek: 
                    output.sim_S_uu = simulations[14] 
                    output.sim_S_dd = simulations[15] 
                    output.sim_sigma_dd = simulations[17] 
                    output.sim_sigma_uu = simulations[16] 
        return output
           



    