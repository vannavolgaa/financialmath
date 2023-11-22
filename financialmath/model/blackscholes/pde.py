from dataclasses import dataclass


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
    initial : np.array 
    spot_vector : np.array 
    vol_up : np.array = None
    vol_down : np.array = None
    r_up : np.array = None
    q_up : np.array = None 
    ds : float = 0.01
    dv : float = 0.01
    dr : float = 0.01 
    dq : float = 0.01 
    dt : float = 0.01

class PDEBlackScholes: 

    def __init__(self, inputdata:PDEBlackScholesInput):
        self.inputdata = inputdata 
        self.N = self.inputdata.number_steps
        self.M = self.inputdata.spot_vector_size
        self.dt = self.inputdata.t/self.N
        self.sigma = self.inputdata.sigma 
        self.r = self.inputdata.r 
        self.q = self.inputdata.q 
        self.S = self.inputdata.S 
        self.ds = self.inputdata.ds
        self.dv = self.inputdata.dv
        self.dr = self.inputdata.dr
        self.dq = self.inputdata.dq 
        self.df = 1/(1+r*self.dt)
        self.dx = self.logspot_step()

    def logspot_step(self) -> float: 
        return self.sigma * np.sqrt(2*self.dt)

    @staticmethod
    def generate_spot_vector(dx: float, S: float, M : int) -> np.array: 
        spotvec = np.empty(M)
        spotvec[0] = S*np.exp((-dx*M/2))
        for i in range(1,M): 
            spotvec[i] = spotvec[i-1]*np.exp(dx)
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

    

