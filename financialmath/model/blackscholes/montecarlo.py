from dataclasses import dataclass
from enum import Enum
from typing import List, NamedTuple
import numpy as np
import time
from abc import abstractmethod, ABC
from financialmath.tools.simulation import (
    RandomGenerator, 
    RandomGeneratorType, 
    NormalDistribution)
from financialmath.tools.tool import MainTool

class BlackScholesSimulation(ABC): 

    @abstractmethod
    def spot_simulation(self): 
        pass

@dataclass
class EulerBlackScholesSimulation(BlackScholesSimulation): 
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
        return np.cumprod(
            a = np.exp(self.drift()+self.diffusion()), 
            axis=1)
    
    def spot_simulation(self) -> np.array: 
        return self.S*self.moneyness_simulation()

@dataclass
class MilsteinBlackScholesSimulation(BlackScholesSimulation): 
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
      
class BlackScholesDiscretization(Enum): 
    euler = EulerBlackScholesSimulation
    milstein = MilsteinBlackScholesSimulation

class MonteCarloBlackScholesParameterMapping(Enum): 
    initial = 0 
    S_up = 1
    r_up = 2
    q_up = 3 
    sigma_up = 4
    t_up = 5 
    S_down = 6 
    sigma_down = 7 
    S_up_sigma_up = 8 
    S_down_sigma_up = 9
    S_up_t_up = 10 
    S_down_t_up = 11 
    sigma_up_t_up = 12 
    sigma_down_t_up = 13
    S_uu = 14 
    S_dd = 15 
    sigma_uu = 16 
    sigma_dd = 17
    S_uuu = 18 
    S_ddd = 19

@dataclass
class MonteCarloBlackScholesSimulationList: 
    initial : np.array 
    S_up : np.array = None
    r_up : np.array = None
    q_up : np.array = None
    sigma_up : np.array = None
    t_up : np.array = None
    S_down : np.array = None
    sigma_down : np.array = None
    S_up_sigma_up : np.array = None
    S_down_sigma_up : np.array = None
    S_up_t_up : np.array = None
    S_down_t_up : np.array = None
    sigma_up_t_up : np.array = None
    sigma_down_t_up : np.array = None
    S_uu : np.array = None
    S_dd : np.array = None
    sigma_uu : np.array = None
    sigma_dd : np.array = None
    S_uuu : np.array = None
    S_ddd : np.array = None

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
    randoms_generator : RandomGeneratorType = \
        RandomGeneratorType.antithetic
    discretization: BlackScholesDiscretization = \
        BlackScholesDiscretization.euler
    dS : float = 0.01 
    dsigma : float = 0.01 
    dr : float = 0.01 
    dq : float = 0.01 
    max_workers : bool = 4

    first_order_greek_id_list = [1,2,3,4,5]
    second_order_greek_id_list = [6,7,8,9,10,11,12,13]
    third_order_greek_id_list = [14,15,16,17]

    def __post_init__(self): 
        self.dt = self.t/self.number_steps
        self.dt_up = (self.dt*self.number_steps+self.dt)/self.number_steps

    def parameters_definition(self): 
        s, ds, dt, r, q, dr, dq, S, dS= self.sigma, self.dsigma,\
        self.dt, self.r, self.q,self.dr, self.dq, self.S, self.dS
        return [(S, r, q, s, dt, 0),
                (S+dS, r, q, s, dt, 1), 
                (S, r+dr, q, s, dt, 2),
                (S, r, q+dq, s, dt, 3), 
                (S, r, q, s+ds, dt, 4), 
                (S, r, q, s, self.dt_up, 5),
                (S-dS, r, q, s, dt, 6), 
                (S, r, q, s-ds, dt, 7),
                (S+dS, r, q, s+ds, dt, 8), 
                (S-dS, r, q, s+ds, dt, 9), 
                (S+dS, r, q, s, self.dt_up, 10),
                (S-dS, r, q, s, self.dt_up, 11),
                (S, r, q, s+ds, self.dt_up, 12),
                (S, r, q, s-ds, self.dt_up, 13), 
                (S+2*dS, r, q, s, dt, 14), 
                (S-2*dS, r, q, s, dt, 15),
                (S, r, q, s+2*ds, dt, 16), 
                (S, r, q, s-2*ds, dt, 17),
                (S+3*dS, r, q, s, dt, 18), 
                (S-3*dS, r, q, s, dt, 19)] 
    
    def get_ids(self, greek1: bool, greek2:bool, greek3:bool) -> List[int]: 
        output = [0]
        if greek1: 
            output = output + self.first_order_greek_id_list 
            if greek2: 
                output = output + self.second_order_greek_id_list 
                if greek3: 
                    output = output + self.third_order_greek_id_list 
        return output
    
    def get_simulation_parameters(self,greek1: bool, greek2:bool, greek3:bool)\
        -> List[tuple]:  
        parameters = self.parameters_definition()
        param_ids = self.get_ids(greek1,greek2,greek3) 
        return [p for p in parameters if p[5] in param_ids]

@dataclass
class MonteCarloBlackScholesOutput: 

    simulations : MonteCarloBlackScholesSimulationList
    time_taken : float 
    dS : float = 0.01
    dsigma : float = 0.01
    dr : float = 0.01 
    dq : float = 0.01
    dt : float = 0.01

class MonteCarloBlackScholes: 

    def __init__(self, inputdata: MonteCarloBlackScholesInput): 
        self.start = time.time()
        self.inputdata = inputdata
        self.N, self.M = inputdata.number_steps, inputdata.number_paths
        self.Z = self.generate_randoms()

    def generate_randoms(self) -> np.array:
        N,M = self.N, self.M
        generator =  RandomGenerator(
            probability_distribution=NormalDistribution(), 
            generator_type=self.inputdata.randoms_generator
            )
        randoms = generator.generate(N=M*N)
        return np.reshape(a=randoms, newshape=(M,N))
    
    def get_simulation_names(self,ids:list[int]) -> List[str]: 
        return [l.name for l in list(MonteCarloBlackScholesParameterMapping)
                if l.value in ids]

    def compute_simulation(self, arg:tuple[float]) -> np.array: 
        S,r,q,sigma,dt,id,Z = arg[0],arg[1],arg[2],arg[3],arg[4],arg[5],self.Z
        simulator = self.inputdata.discretization.value(
                S = S, 
                r = r, 
                q = q, 
                sigma = sigma,
                Z = Z,
                dt = dt, 
                future = self.inputdata.future
                )
        return {id:simulator.spot_simulation()}
    
    def get_simulations(self, 
                         first_order_greek: bool=True, 
                         second_order_greek:bool=True, 
                         third_order_greek:bool=True)\
                            -> MonteCarloBlackScholesSimulationList: 
        sim_parameters = self.inputdata.get_simulation_parameters(
            greek1=first_order_greek,
            greek2=second_order_greek,
            greek3=third_order_greek
        )
        simulations = MainTool.send_task_with_futures(
            task=self.compute_simulation,
            args=sim_parameters, 
            max_workers=self.inputdata.max_workers
            )
        result = MainTool.listdict_to_dictlist(simulations)
        sim_names = self.get_simulation_names(list(result.keys()))
        result = dict(zip(sim_names, list(result.values())))
        return MonteCarloBlackScholesSimulationList(**result)

    def get(self, first_order_greek:bool = True, 
            second_order_greek:bool = True, 
            third_order_greek:bool = True) -> MonteCarloBlackScholesOutput: 
        simulations = self.get_simulations(first_order_greek,
                                           second_order_greek,
                                           third_order_greek)
        return MonteCarloBlackScholesOutput(
            simulations=simulations,
            time_taken=time.time()-self.start, 
            dS = self.inputdata.dS, 
            dsigma = self.inputdata.dsigma, 
            dt = self.inputdata.dt, 
            dr = self.inputdata.dr, 
            dq=self.inputdata.dq)
           



    