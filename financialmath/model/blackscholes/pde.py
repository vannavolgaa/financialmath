from dataclasses import dataclass
import numpy as np
from typing import List, NamedTuple
from scipy import sparse
import time
from financialmath.tools.finitedifference import OneFactorImplicitScheme
from financialmath.tools.tool import MainTool

class PDEBlackScholesInput(NamedTuple): 
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
    max_workers = 7 

@dataclass
class PDEBlackScholesOutput: 
    grid_list : np.array 
    spot_vector : np.array
    step_vector : np.array 
    t_vector : np.array 
    time_taken : float
    grid_list_sigma_up : np.array = None
    grid_list_sigma_down : np.array = None
    grid_list_r_up : np.array = None
    grid_list_q_up : np.array = None
    grid_list_sigma_uu : np.array = None 
    grid_list_sigma_dd : np.array = None
    dS : float = 0.01
    dsigma : float = 0.01
    dr : float = 0.01 
    dq : float = 0.01 
    dt : float = 0.01
    
@dataclass 
class ImplicitBlackScholes: 
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
        self.dS = self.inputdata.dS
        self.ds = self.inputdata.dsigma
        self.dr = self.inputdata.dr
        self.dq = self.inputdata.dq 
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
    
    def compute_matrixes(self, arg:tuple[float])\
        -> dict[int,List[sparse.csc_matrix]]:
        r, q, sigma, id= arg[0], arg[1], arg[2], arg[3]
        scheme = ImplicitBlackScholes(
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
    
    def args_fd_list(self, greek1:bool = True, 
            greek2:bool = True, greek3:bool = True) -> List[tuple]: 
        s, ds, r, q, dr, dq,= self.sigma, self.ds, self.r, self.q,\
            self.dr, self.dq
        args_list = [(r, q, s, 0)]
        if greek1: 
            new_args = [(r+dr, q, s, 1),
                        (r, q+dq, s, 2), 
                        (r, q, s+ds, 3)] 
            args_list = args_list+new_args
            if greek2: 
                new_args = [(r, q, s-ds, 4)] 
                args_list = args_list+new_args 
                if greek3: 
                    new_args = [(r, q, s-2*ds, 5),
                                (r, q, s+2*ds, 6)] 
                    args_list = args_list+new_args
        return args_list

    def get_matrixes(self,greek1:bool = True, greek2:bool = True, 
            greek3:bool = True) -> dict[int, np.array]: 
        args = self.args_fd_list(greek1,greek2,greek3)
        matrixes = MainTool.send_task_with_futures(
            task = self.compute_matrixes,
            args = args, 
            max_workers=self.inputdata.max_workers
            )
        return MainTool.listdict_to_dictlist(matrixes)
    
    def get(self, first_order_greek:bool = True, 
            second_order_greek:bool = True, 
            third_order_greek:bool = True) -> PDEBlackScholesOutput: 
        matrixes = self.get_matrixes(first_order_greek, 
                                     second_order_greek, 
                                     third_order_greek)
        output = PDEBlackScholesOutput(
            grid_list = matrixes[0], 
            t_vector=self.t_vector(), 
            step_vector=self.step_vector(),
            spot_vector=self.spot_vector(),
            time_taken=time.time()-self.start, 
            dS = self.dS, 
            dsigma = self.ds, 
            dt = self.dt, 
            dr = self.dr, 
            dq=self.dq
            )
        if first_order_greek: 
            output.grid_list_sigma_up = matrixes[3] 
            output.grid_list_r_up = matrixes[1] 
            output.grid_list_q_up = matrixes[2] 
            if second_order_greek: 
                output.grid_list_sigma_down = matrixes[4] 
                if third_order_greek: 
                    output.grid_list_sigma_uu = matrixes[6] 
                    output.grid_list_sigma_dd = matrixes[5] 
        return output


    

