from dataclasses import dataclass
import numpy as np
from typing import List, NamedTuple
from scipy import sparse
import time
from enum import Enum
from financialmath.tools.finitedifference import OneFactorImplicitScheme
from financialmath.tools.tool import MainTool

@dataclass 
class PDEImplicitBlackScholes: 
    sigma : float 
    r : float 
    q : float 
    dx : float 
    dt : float 
    M : int 
    N : int 
    future: bool

    def __post_init__(self): 
        self.df = 1/(1+self.r*self.dt)
        self.p = self.probability()
    
    def probability(self) -> float : 
        sigma, dt, dx= self.sigma, self.dt, self.dx
        return dt*(sigma**2)/(2*(dx**2))

    def up_move(self) -> float: 
        sigma, dt, dx, r, q = self.sigma, self.dt, self.dx, self.r, self.q
        if self.future: r,q = 0,0
        u = (r-q-(sigma**2)/2)*dt/(2*dx)
        pu = self.df*(self.p + u)
        return np.reshape(
            a = np.repeat(pu,self.M*self.N),
            newshape = (self.M,self.N)
            ) 

    def down_move(self) -> float:
        sigma, dt, dx, r, q = self.sigma, self.dt, self.dx, self.r, self.q
        if self.future: r,q = 0,0
        d = (r-q-(sigma**2)/2)*dt/(2*dx)
        pd = self.df*(self.p - d)  
        return np.reshape(
            a = np.repeat(pd,self.M*self.N),
            newshape = (self.M,self.N)
            )  

    def mid_move(self) -> float: 
        pm = self.df*(1-2*self.p)
        return np.reshape(
            a = np.repeat(pm,self.M*self.N),
            newshape = (self.M,self.N)
            ) 
    
    def transition_matrixes(self) -> List[sparse.csc_matrix]: 
        scheme = OneFactorImplicitScheme(
            up_matrix = self.up_move(), 
            down_matrix = self.down_move(), 
            mid_matrix = self.mid_move(),
            N = self.N, 
            M = self.M
            )
        return scheme.transition_matrixes()

class PDEBlackScholesParameterMapping(Enum): 
    initial = 0 
    r_up = 1
    q_up = 2
    sigma_up = 3
    sigma_down = 4
    sigma_uu = 5
    sigma_dd = 6

@dataclass
class PDEBlackScholesMatrixes: 
    initial : np.array 
    r_up : np.array = None
    q_up : np.array = None
    sigma_up : np.array = None
    sigma_down : np.array = None
    sigma_uu : np.array = None
    sigma_dd : np.array = None

@dataclass
class PDEBlackScholesInput: 
    S : float 
    r : float 
    q : float 
    sigma : float 
    t : float 
    number_steps : int = 400
    spot_vector_size : int = 100
    future : bool = False
    dS : float = 0.01 
    dsigma : float = 0.01 
    dr : float = 0.01 
    dq : float = 0.01 
    max_workers : int = 7 

    first_order_greek_id_list = [1,2,3]
    second_order_greek_id_list = [4,5]
    third_order_greek_id_list = [6,7]

    def parameters_definition(self): 
        s, ds, r, q, dr, dq= self.sigma, self.dsigma,self.r,\
            self.q,self.dr, self.dq
        return [(r, q, s, 0),
                (r+dr, q, s, 1), 
                (r, q+dq, s, 2),
                (r, q, s+ds, 3),
                (r, q, s-ds, 4), 
                (r, q, s-2*ds, 5), 
                (r, q, s+2*ds, 6)] 

    def get_ids(self, greek1: bool, greek2:bool, greek3:bool) -> List[int]: 
        output = [0]
        if greek1: 
            output = output + self.first_order_greek_id_list 
            if greek2: 
                output = output + self.second_order_greek_id_list 
                if greek3: 
                    output = output + self.third_order_greek_id_list 
        return output
    
    def get_pde_parameters(self,greek1: bool, greek2:bool, greek3:bool)\
        -> List[tuple]:  
        parameters = self.parameters_definition()
        param_ids = self.get_ids(greek1,greek2,greek3) 
        return [p for p in parameters if p[3] in param_ids]

@dataclass
class PDEBlackScholesOutput: 
    matrixes : PDEBlackScholesMatrixes
    spot_vector : np.array
    time_taken : float
    dS : float = 0.01
    dsigma : float = 0.01
    dr : float = 0.01 
    dq : float = 0.01 
    dt : float = 0.01

class PDEBlackScholes: 

    def __init__(self, inputdata:PDEBlackScholesInput):
        self.start = time.time()
        self.inputdata = inputdata 
        self.N = self.inputdata.number_steps
        self.M = self.inputdata.spot_vector_size
        self.dt = self.inputdata.t/self.N
        self.sigma = self.inputdata.sigma 
        self.r = self.inputdata.r 
        self.q = self.inputdata.q 
        self.S = self.inputdata.S 
        self.df = 1/(1+self.r*self.dt)
        self.dx = self.logspot_step()

    def logspot_step(self) -> float: 
        return self.sigma * np.sqrt(2*self.dt)
    
    def t_vector(self) -> np.array: 
        return np.cumsum(np.repeat(self.dt, self.N))
    
    def step_vector(self) -> np.array: 
        return np.cumsum(np.repeat(1, self.N))

    def spot_vector(self) -> np.array: 
        spotvec = np.empty(self.M)
        spotvec[0] = self.S*np.exp((-self.dx*self.M/2))
        for i in range(1,self.M): 
            spotvec[i] = spotvec[i-1]*np.exp(self.dx)
        return spotvec
    
    def get_matrixes_names(self,ids:list[int]) -> List[str]: 
        return [l.name for l in list(PDEBlackScholesParameterMapping)
                if l.value in ids]

    def compute_matrixes(self, arg:tuple[float])\
        -> dict[int,List[sparse.csc_matrix]]:
        r, q, sigma, id= arg[0], arg[1], arg[2], arg[3]
        scheme = PDEImplicitBlackScholes(
            sigma = sigma,
            r = r,
            q = q,
            dx = self.dx,
            dt = self.dt,
            M = self.M,
            N = self.N, 
            future = self.inputdata.future
            )
        return {id:scheme.transition_matrixes()}
    
    def get_matrixes(self,greek1:bool = True, greek2:bool = True, 
            greek3:bool = True) -> PDEBlackScholesMatrixes: 
        pde_parameters = self.inputdata.get_pde_parameters(
            greek1=greek1, greek2=greek2, greek3=greek3
        )
        matrixes = MainTool.send_task_with_futures(
            task = self.compute_matrixes,
            args = pde_parameters, 
            max_workers=self.inputdata.max_workers
            )
        result = MainTool.listdict_to_dictlist(matrixes)
        sim_names = self.get_matrixes_names(list(result.keys()))
        result = dict(zip(sim_names, list(result.values())))
        return PDEBlackScholesMatrixes(**result)

    def get(self, first_order_greek:bool = True, 
            second_order_greek:bool = True, 
            third_order_greek:bool = True) -> PDEBlackScholesOutput: 
        return PDEBlackScholesOutput(
            matrixes = self.get_matrixes(first_order_greek, 
                                     second_order_greek, 
                                     third_order_greek),
            spot_vector=self.spot_vector(),
            time_taken=time.time()-self.start, 
            dS = self.inputdata.dS, 
            dsigma = self.inputdata.dsigma, 
            dt = self.dt, 
            dr = self.inputdata.dr, 
            dq=self.inputdata.dq
            )

    

