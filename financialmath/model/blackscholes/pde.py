from dataclasses import dataclass
import numpy as np

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
    greeks : bool = True
    ds : float = 0.01 
    dv : float = 0.01 
    dr : float = 0.01 
    dq : float = 0.01 

@dataclass
class PDEBlackScholesOutput: 
    grid_list : np.array 
    spot_vector : np.array
    step_vector : np.array 
    t_vector : np.array 
    grid_list_sigma_up : np.array = None
    grid_list_sigma_down : np.array = None
    grid_list_r_up : np.array = None
    grid_list_q_up : np.array = None
    grid_list_sigma_uu : np.array = None 
    grid_list_sigma_dd : np.array = None
    ds : float = 0.01
    dv : float = 0.01
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

    def __post_init__(self): 
        self.df = 1/(1+self.r*self.dt)
        self.p = self.probability()
    
    def probability(self) -> float : 
        sigma, dt, dx= self.sigma, self.dt, self.dx
        return dt*(sigma**2)/(2*(dx**2))

    def up_move(self) -> float: 
        sigma, dt, dx, r, q = self.sigma, self.dt, self.dx, self.r, self.q
        u = (r-q-(sigma**2)/2)*dt/(2*dx)
        pu = self.df*(self.p + u)
        return np.reshape(np.repeat(pu,M*N),(M,N)) 

    def down_move(self, sigma:float, r:float, q:float) -> float:
        sigma, dt, dx, r, q = self.sigma, self.dt, self.dx, self.r, self.q
        d = (r-q-(sigma**2)/2)*dt/(2*dx)
        pd = self.df*(self.p - d)  
        return np.reshape(np.repeat(pd,M*N),(M,N))  

    def mid_move(self, sigma:float) -> float: 
        pm = self.df*(1-2*self.p)
        return np.reshape(np.repeat(pm,M*N),(M,N))

class PDEBlackScholes: 

    def __init__(self, inputdata:PDEBlackScholesInput):
        self.inputdata = inputdata 
        self.N = self.inputdata.number_steps
        self.M = self.inputdata.spot_vector_size
        self.tau = self.inputdata.T - self.inputdata.t
        self.dt = self.tau/self.N
        self.sigma = self.inputdata.sigma 
        self.r = self.inputdata.r 
        self.q = self.inputdata.q 
        self.S = self.inputdata.S 
        self.ds = self.inputdata.ds
        self.dv = self.inputdata.dv
        self.dr = self.inputdata.dr
        self.dq = self.inputdata.dq 
        self.df = 1/(1+self.r*self.dt)
        self.dx = self.logspot_step()

    def logspot_step(self) -> float: 
        return self.sigma * np.sqrt(2*self.dt)

    def generate_spot_vector(self) -> np.array: 
        spotvec = np.empty(self.M)
        spotvec[0] = self.S*np.exp((-self.dx*self.M/2))
        for i in range(1,self.M): 
            spotvec[i] = spotvec[i-1]*np.exp(self.dx)
        return spotvec
    
    def discounted_probability_up_move(self, sigma:float, r:float, q:float) -> float: 
        p = self.dt*(sigma**2)/(2*(self.dx**2))
        u = (r-q-(sigma**2)/2)*self.dt/(2*self.dx)
        return np.reshape(np.repeat(self.df*(p + u),self.M*self.N),(self.M,self.N)) 

    def discounted_probability_down_move(self, sigma:float, r:float, q:float) -> float:
        p = self.dt*(sigma**2)/(2*(self.dx**2))
        d = (r-q-(sigma**2)/2)*self.dt/(2*self.dx)
        return self.df*(p - d)  

    def discounted_probability_mid_move(self, sigma:float) -> float: 
        p = self.dt*(sigma**2)/(2*(self.dx**2))
        return self.df*(1-2*p)

    

