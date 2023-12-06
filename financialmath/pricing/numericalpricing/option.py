import numpy as np 
from dataclasses import dataclass
from scipy import sparse, interpolate
from typing import List
from financialmath.instruments.option import *

@dataclass
class OptionPayOffTool: 
    spot:np.array or float
    strike: np.array or float
    barrier_up: np.array or float
    barrier_down: np.array or float
    gap_trigger: np.array or float
    binary_amount: np.array or float
    rebate: np.array or float
    payoff : OptionPayoff

    def barrier_condition(self) -> np.array or bool: 
        S, b_up, b_down=self.spot, self.barier_up, self.barrier_down
        match self.payoff.barrier_type: 
            case BarrierType.up_and_in: condition = (S>b_up)
            case BarrierType.up_and_out: condition = (S<b_up)
            case BarrierType.down_and_in: condition = (S<b_down)
            case BarrierType.down_and_out: condition = (S>b_down)
            case BarrierType.double_knock_in: condition = (S<b_down)&(S>b_up)
            case BarrierType.double_knock_out: condition = (S>b_down)&(S<b_up)
        return condition.astype(int) 
    
    def rebate_payoff(self) -> np.array or float: 
        barrier_cond = self.barrier_condition()
        barrier_invcond = np.abs(np.array(barrier_cond) -1)
        return barrier_invcond*self.rebate

    def vanilla_payoff(self) -> np.array or float: 
        S,K = self.spot, self.strike
        match self.payoff.option_type: 
            case OptionalityType.call: return np.maximum(S-K,0)
            case OptionalityType.put: return np.maximum(K-S,0)

    def gap_payoff(self) -> np.array or float: 
        S, G, K = self.spot,self.gap_trigger, self.strike
        match self.payoff.option_type: 
            case OptionalityType.call: return (S>G).astype(int)*(S-K)
            case OptionalityType.put: return (S<G).astype(int)*(K-S)
    
    def binary_payoff(self, payoff: np.array) -> np.array or float:
        return (abs(payoff)>0).astype(int)*self.binary_amount*np.sign(payoff)
    
    def payoff_vector(self) -> np.array or float: 
        if self.payoff.gap: payoff = self.gap_payoff()
        else: payoff = self.vanilla_payoff()
        if self.payoff.is_barrier(): 
            barrier_cond = self.barrier_condition()
            rebate_payoff = self.rebate_payoff()
            payoff = payoff * barrier_cond + rebate_payoff
        if self.payoff.binary: payoff = self.binary_payoff(payoff = payoff)
        return payoff

@dataclass
class OneFactorOptionPriceGrids: 
    initial : np.array 
    spot_vector : np.array
    S : float 
    vol_up : np.array = None
    vol_down : np.array = None
    vol_uu : np.array = None
    vol_dd : np.array = None
    r_up : np.array = None
    q_up : np.array = None
    dt : float = 0
    dS: float = 0.01
    dsigma: float = 0.01
    dr : float = 0.01
    dq : float = 0.01

@dataclass
class OneFactorFiniteDifferenceGreeks: 

    inputdata : OneFactorOptionPriceGrids 
    interpolation_method: str = 'cubic'

    def __post_init__(self): 
        self.S = self.inputdata.S
        self.dt = self.inputdata.dt
        self.dS = self.inputdata.dS
        self.dsigma = self.inputdata.dsigma
        self.dr = self.inputdata.dr
        self.dq = self.inputdata.dq
        self.spot_vector = self.inputdata.spot_vector
        self.V_0 = self.read_grid(self.inputdata.initial, 0)
        self.V_dt = self.read_grid(self.inputdata.initial, 1)
        self.V_0_rp = self.read_grid(self.inputdata.r_up, 0)
        self.V_0_qp = self.read_grid(self.inputdata.q_up, 0)
        self.V_0_volu = self.read_grid(self.inputdata.vol_up, 0)
        self.V_0_vold = self.read_grid(self.inputdata.vol_down, 0)
        self.V_dt_volu = self.read_grid(self.inputdata.vol_up, 1)
        self.V_0_voluu = self.read_grid(self.inputdata.vol_uu, 0)
        self.V_0_voldd = self.read_grid(self.inputdata.vol_dd, 0)
        
    def read_grid(self, grid:np.array, pos: int) -> np.array: 
        try: return grid[:,pos]
        except TypeError: return np.repeat(np.nan, len(self.spot_vector))

    def interpolate_value(self, value:float, x: np.array, y:np.array) -> float:
        try: 
            f = interpolate.interp1d(x=x, y=y, kind = self.interpolation_method)
            return f(value).item()
        except: return np.nan 
    
    def option_price(self, S: float, vector: np.array) -> float: 
        return self.interpolate_value(S, self.spot_vector, vector)
    
    def price(self) -> float: 
        return self.option_price(self.S, self.V_0) 
    
    def delta(self, S:float, vector: np.array, h: float) -> float: 
        Su = S*(1+h)
        Sd = S*(1-h)
        Vu = self.option_price(S=Su, vector=vector)
        Vd = self.option_price(S=Sd, vector=vector)
        return (Vu-Vd)/(Su-Sd)

    def gamma(self, S:float, vector: np.array, h: float) -> float: 
        Su = S*(1+h)
        Sd = S*(1-h)
        delta_up = self.delta(S=Su, vector=vector, h=h)
        delta_down = self.delta(S=Sd, vector=vector, h=h)
        return (delta_up-delta_down)/(Su-Sd)

    def speed(self, S:float, vector: np.array, h: float) -> float: 
        Su = S*(1+h)
        Sd = S*(1-h)
        gamma_up = self.gamma(S=Su, vector=vector, h=h)
        gamma_down = self.gamma(S=Sd, vector=vector, h=h)
        return (gamma_up-gamma_down)/(Su-Sd)

    def theta(self, S:float, uvec: np.array, 
            dvec: np.array, h:float) -> float:  
        Vu = self.option_price(S=S, vector=uvec)
        Vd = self.option_price(S=S, vector=dvec)
        return (Vu-Vd)/h

    def vega(self, S:float, uvec: np.array, 
            dvec: np.array, h:float) -> float:  
        Vu = self.option_price(S=S, vector=uvec)
        Vd = self.option_price(S=S, vector=dvec)
        return (Vu-Vd)/h
    
    def rho(self, S:float, uvec: np.array, 
            dvec: np.array, h:float) -> float:  
        Vu = self.option_price(S=S, vector=uvec)
        Vd = self.option_price(S=S, vector=dvec)
        return (Vu-Vd)/h

    def epsilon(self, S:float, uvec: np.array, 
            dvec: np.array, h:float) -> float:  
        Vu = self.option_price(S=S, vector=uvec)
        Vd = self.option_price(S=S, vector=dvec)
        return (Vu-Vd)/h

    def vanna(self, S:float, uvec: np.array, dvec: np.array, 
            h_S:float, h_vol:float) -> float: 
        delta_up = self.delta(S=self.S, vector=uvec, h=h_S)
        delta_down = self.delta(S=self.S, vector=dvec, h=h_S)
        return (delta_up-delta_down)/h_vol

    def volga(self, S:float, uvec: np.array, 
            vec: np.array, dvec: np.array, h:float): 
        Vu = self.option_price(S=S, vector=uvec)
        V = self.option_price(S=S, vector=vec)
        Vd = self.option_price(S=S, vector=dvec)
        return (Vu+Vd -2*V)/(h**2)

    def charm(self, S:float, uvec: np.array, 
            dvec: np.array, h_S:float, dt:float): 
        delta_up = self.delta(S=S, vector=uvec, h=h_S)
        delta_down = self.delta(S=S, vector=dvec, h=h_S)
        return (delta_up-delta_down)/dt

    def veta(self, S:float, uvec_dt: np.array, dvec_dt: np.array, 
            uvec: np.array, dvec: np.array, h_vol: float, dt:float): 
        vega_up = self.vega(S=S, uvec=uvec_dt, dvec=dvec_dt, h=h_vol) 
        vega_down = self.vega(S=S, uvec=uvec, dvec=dvec, h=h_vol) 
        return (vega_up-vega_down)/dt

    def zomma(self, S:float, uvec: np.array, dvec: np.array, 
            h_S: float, h_vol: float): 
        gamma_up = self.gamma(S=S, vector=uvec, h=h_S)
        gamma_down = self.gamma(S=S, vector=dvec, h=h_S)
        return (gamma_up-gamma_down)/h_vol
    
    def color(self, S:float, uvec: np.array, dvec: np.array, 
            h_S: float, dt: float): 
        gamma_up = self.gamma(S=S, vector=uvec, h=h_S)
        gamma_down = self.gamma(S=S, vector=dvec, h=h_S)
        return (gamma_up-gamma_down)/dt
    
    def vera(self) -> float: 
        return np.nan

    def ultima(self) -> float: 
        return np.nan
    
    def greeks(self) -> dict[str, float]: 
        return {
            'delta':self.delta(self.S,self.V_0,self.dS), 
            'vega':self.vega(self.S,self.V_0_volu,self.V_0,self.dsigma), 
            'theta':self.theta(self.S,self.V_dt,self.V_0,self.dt),
            'rho':self.rho(self.S,self.V_0_rp,self.V_0,self.dr),
            'epsilon':self.epsilon(self.S,self.V_0_qp,self.V_0,self.dq),
            'gamma':self.gamma(self.S, self.V_0, self.dS),
            'vanna':self.vanna(self.S, self.V_0_volu,self.V_0,self.dS,
                             self.dsigma),
            'volga':self.volga(self.S,self.V_0_volu,self.V_0,
                             self.V_0_vold,self.dsigma),
            'charm':self.charm(self.S,self.V_dt,self.V_0,
                             self.dS,self.dt),
            'veta':self.veta(self.S,self.V_dt_volu,self.V_dt,self.V_0_volu,
                           self.V_0,self.dsigma,self.dt),
            'vera':np.nan,
            'speed':self.speed(self.S,self.V_0,self.dS),
            'zomma':self.zomma(self.S,self.V_0_volu,self.V_0,self.dS,
                             self.dsigma),
            'color':self.color(self.S, self.V_dt, self.V_0, self.dS, 
                             self.dt),
            'ultima':np.nan}
        
@dataclass
class OneFactorFiniteDifferencePricer: 
    option : Option 
    grid_list : List[sparse.csc_matrix]
    spot_vector : np.array
    S : float 
    dt : float 
    dS : float 
    interpolation_method: str = 'cubic'
    
    def __post_init__(self): 
        self.spec = self.option.specification
        self.N = len(self.grid_list)
        self. M = len(self.spot_vector)
        self.option_steps = self.spec.get_steps(self.N)
        self.dt = self.spec.tenor.expiry/self.N
        self.ptool = OptionPayOffTool(
            spot=self.spot_vector, 
            strike = self.spec.strike, 
            barrier_down=self.spec.barrier_down, 
            barier_up=self.spec.barrier_up, 
            rebate=self.spec.rebate, 
            gap_trigger=self.spec.gap_trigger,
            binary_amount=self.spec.binary_amout,
            payoff = self.option.payoff)
    
    def interpolate_value(self, value:float, x: np.array, y:np.array) -> float:
        try: 
            f = interpolate.interp1d(x=x, y=y, kind = self.interpolation_method)
            return f(value).item()
        except: return np.nan 

    def early_exercise_condition(self, ptool: OptionPayOffTool,
                                  prices : np.array, n:int) -> np.array: 
        payoff = ptool.payoff_vector()
        match self.option.payoff.exercise:
            case ExerciseType.european: return prices
            case ExerciseType.american: return np.maximum(payoff, prices)
            case ExerciseType.bermudan: 
                if n in self.option_steps.bermudan: 
                    return np.maximum(payoff, prices)
                else: return prices
    
    def touch_barrier_condition(self, ptool: OptionPayOffTool,
                                prices: np.array, n:int) -> np.array: 
        condition = ptool.barrier_condition()
        match self.option.payoff.barrier_observation: 
            case ObservationType.continuous: 
                return condition*prices + self.ptool.rebate_payoff()
            case ObservationType.in_fine: return prices
            case ObservationType.discrete: 
                if n in self.option_steps.barrier_discrete: 
                    return condition*prices + self.ptool.rebate_payoff()
                else: return prices
            case ObservationType.window:
                end = self.option_steps.barrier_window_end
                begin = self.option_steps.barrier_window_begin
                if n >= begin and n <= end: 
                    return condition*prices+self.ptool.rebate_payoff()
                else: return prices
            case _: return prices
            
    def check_path_condition(self, ptool: OptionPayOffTool, 
                             prices : np.array, n:int) -> np.array: 
        if ptool.payoff.is_barrier():
            prices = self.touch_barrier_condition(ptool,prices, n)
        if ptool.payoff.is_early_exercise():
            prices = self.early_exercise_condition(ptool, prices, n)
        return prices
    
    def generate_grid(self, ptool: OptionPayOffTool) -> np.array: 
        p0 = ptool.payoff_vector()
        grid_shape = (self.M, self.N)
        grid = np.zeros(grid_shape)
        grid[:, self.N-1] = p0
        price_vec = p0
        for n in range(self.N-1, -1, -1):
            tmat =  self.grid_list[n]
            price_vec = tmat.dot(price_vec) 
            price_vec = self.check_path_condition(ptool,price_vec,n)
            grid[:, n] = price_vec
        return grid 

    def grid_fallback(self): 
        M, N = self.M, self.N
        return np.reshape(np.repeat(np.nan, M*N), (M,N)) 

    def vanilla_grid(self) -> np.array:
        option_payoff = self.option.payoff
        payoff = OptionPayoff(
            exercise=option_payoff.exercise, 
            binary=option_payoff.binary, 
            option_type=option_payoff.option_type, 
            gap =option_payoff.gap)
        ptool = self.ptool
        ptool.payoff = payoff
        return self.generate_grid(ptool)
    
    def barrier_out_grid(self, barrier: BarrierType) -> np.array: 
        option_payoff = self.option.payoff
        payoff = OptionPayoff(
            exercise=option_payoff.exercise, 
            binary=option_payoff.binary, 
            option_type=option_payoff.option_type, 
            gap =option_payoff.gap, 
            barrier_observation=option_payoff.barrier_observation, 
            barrier_type=barrier)
        ptool = self.ptool
        ptool.payoff = payoff
        return self.generate_grid(ptool)
    
    def barrier_in_grid(self) -> np.array:
        barrier = self.option.payoff.get_opposite_barrier()
        vanilla = self.vanilla_grid()
        barrier_out = self.barrier_out_grid(barrier = barrier)
        return vanilla - barrier_out
    
    def option_price_grid(self) -> np.array: 
        option_payoff = self.option.payoff
        if option_payoff.forward_start: return self.grid_fallback()
        if option_payoff.is_lookback(): return self.grid_fallback()
        if option_payoff.is_barrier(): 
            if option_payoff.is_in_barrier(): return self.barrier_in_grid() 
            else: return self.generate_grid(self.ptool)
        else: return self.generate_grid(self.ptool)

@dataclass
class MonteCarloGreeks: 
    S: float 
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
    
    def __post_init__(self): 
        self.Su = self.S*(1+self.dS)
        self.Suu = self.S*(1+self.dS)**2
        self.Sdd = self.S*(1-self.dS)**2
        self.Sd = self.S*(1-self.dS)
        

    def price(self): 
        return self.V

    #first order
    def delta(self): 
        return (self.V_S_u-self.V)/(self.Su-self.S)
    
    def vega(self): 
        return (self.V_sigma_u-self.V)/self.dsigma 
    
    def theta(self): 
        return (self.V_t_u-self.V)/-self.dt

    def rho(self): 
        return (self.V_r_u-self.V)/self.dr

    def epsilon(self): 
        return (self.V_q_u-self.V)/self.dq
    
    # second order   
    def gamma(self): 
        return (self.V_S_u+self.V_S_d-2*self.V)/((100*self.dS)**2)
    
    def volga(self): 
        return (self.V_sigma_u+self.V_sigma_d-2*self.V)/(2*self.dsigma) 
    
    def vanna(self): 
        delta_up = (self.V_sigma_u_S_u-self.V)/(self.Su-self.S)
        delta_down = (self.V-self.V_sigma_u_S_d)/(self.S-self.Sd)
        return (delta_up-delta_down)/self.dsigma
    
    def charm(self): 
        delta_up = (self.V_t_u_S_u-self.V)/(self.Su-self.S)
        delta_down = (self.V-self.V_t_u_S_d)/(self.S-self.Sd)
        return (delta_up-delta_down)/-self.dt 

    def veta(self): 
        vega_up = (self.V_t_u_sigma_u-self.V)/self.dsigma 
        vega_down = (self.V-self.V_t_u_sigma_d)/self.dsigma 
        return (vega_up-vega_down)/-self.dt

    #third order
    def speed(self): 
        delta_up = (self.V_S_u-self.V)/(self.Su-self.S)
        delta_down = (self.V-self.V_S_d)/(self.S-self.Sd)
        delta_uu = (self.V_S_uu-self.V_S_u)/(self.Suu-self.Su)
        delta_dd = (self.V_S_d - self.V_S_dd)/(self.Sd-self.Sdd)
        gamma_up = (delta_uu - delta_up)/(self.Suu-self.Su)
        gamma_down = (delta_down - delta_dd)/(self.Sd-self.Sdd)
        return (gamma_up-gamma_down)/(self.Su-self.Sd)
    
    def ultima(self): 
        vega_up = (self.V_sigma_u-self.V)/self.dsigma
        vega_down = (self.V-self.V_sigma_d)/self.dsigma
        vega_uu = (self.V_sigma_uu-self.V_sigma_u)/self.dsigma
        vega_dd = (self.V_sigma_d - self.V_sigma_dd)/self.dsigma
        volga_up = (vega_uu - vega_up)/self.dsigma
        volga_down = (vega_down - vega_dd)/self.dsigma
        return (volga_up-volga_down)/self.dsigma

    def color(self): 
        d = (self.dt*(self.Su-self.Sd)**2)
        return (self.V_t_u_S_u+self.V_t_u_S_d-2*self.V_t_u)/d

    def zomma(self): 
        d = (self.dsigma*(self.Su-self.Sd)**2)
        return (self.V_sigma_u_S_u+self.V_sigma_u_S_d-2*self.V_sigma_u)/d
    
    def greeks(self) -> dict[str,float]: 
        return {'delta':self.delta(), 'vega':self.vega(), 
                'theta':self.theta(), 'rho' : self.rho(),
                'epsilon':self.epsilon(),'gamma':self.gamma(), 
                'vanna':self.vanna(), 'charm':self.charm(), 
                'veta':self.veta(), 'volga':self.volga(),
                'vera':np.nan, 'speed':self.speed(), 
                'zomma':self.zomma(), 'color':self.color(), 
                'ultima':self.ultima()} 

@dataclass
class MonteCarloLookback: 
    sim : np.array 
    forward_start : bool 
    lookback_method : LookbackMethod
    observation_type : ObservationType
    discrete_steps : List[int]
    begin_window_step : int 
    end_window_step : int
    forward_start_step : int
    

    def __post_init__(self): 
        self.N = self.option_steps.N
        self.M = self.sim.shape[0]
        self.fstart_step = self.forward_start_step
    
    @staticmethod
    def progressive_npfun(vectors:np.array, fun:object) -> np.array: 
        emptyvec = np.zeros(vectors.shape)
        emptyvec[:,0] = vectors[:,0]
        for i in range(1,emptyvec.shape[1]): 
            emptyvec[:,i] = fun(vectors[:,0:i], axis=1)
        return emptyvec
    
    @staticmethod
    def vecrep_to_mat(vec:np.array, M:int, N:int) -> np.array: 
        return np.reshape(np.repeat(vec, N), (M,N))
    
    def compute_lookback_method(self, vectors:np.array) -> np.array: 
        match self.lookback_method: 
            case LookbackMethod.geometric_mean: 
                return np.log(self.progressive_npfun(np.exp(vectors),np.mean))
            case LookbackMethod.arithmetic_mean:
                return self.progressive_npfun(vectors,np.mean) 
            case LookbackMethod.minimum:
                return self.progressive_npfun(vectors,np.min) 
            case LookbackMethod.maximum: 
                return self.progressive_npfun(vectors,np.max) 
    
    def continuous_observation(self)-> np.array:
        if self.forward_start:sim = self.sim[:,self.fstart_step:self.N] 
        else: sim = self.sim
        return self.compute_lookback_method(sim)
    
    def window_observation(self) -> np.array: 
        s = self.begin_window_step
        e = self.end_window_step
        out = np.zeros(self.sim.shape)
        out[:,0:s] = self.sim[:,0:s]
        out[:,s:e] = self.compute_lookback_method(self.sim[:,s:e])
        out[:,e:N] = self.vecrep_to_mat(out[:,e-1],self.M,N - e)
        if self.forward_start: return out[:,self.fstart_step:self.N]
        else: return out
    
    def discrete_observation(self) -> np.array: 
        obs, N, M = self.discrete_steps, self.N, self.M
        n_obs, lb = len(obs), self.compute_lookback_method(self.sim[:,obs])
        out = np.zeros(self.sim.shape)
        out[:,0:obs[0]] = self.sim[:,0:obs[0]]
        for i in range(1,n_obs): 
            n=obs[i]-obs[i-1]
            out[:,obs[i-1]:obs[i]]=self.vecrep_to_mat(lb[:,i-1],M,n)
        n = N - obs[n_obs-1]
        out[:,obs[n_obs-1]:N] = self.vecrep_to_mat(lb[:,n_obs-1],self.M,n)
        if self.forward_start: return out[:,self.fstart_step:self.N]
        else: return out
    
    def compute(self) -> np.array: 
        match self.observation_type: 
            case ObservationType.continuous: 
                return self.continuous_observation()
            case ObservationType.discrete: 
                return self.discrete_observation()
            case ObservationType.window: 
                return self.window_observation()
       
@dataclass
class MonteCarloPricing: 
    sim : np.array 
    option : Option 
    r : float

    def __post_init__(self): 
        self.M, self.N = self.sim.shape[0], self.sim.shape[1]
        self.option_steps = self.option.specification.get_steps(self.N)
        self.fstart_step = self.option_steps.forward_start
        self.n_forward_start = self.N - self.fstart_step

    def spot_simulation(self) -> np.array: 
        if self.option.payoff.forward_start: 
            return self.sim[:,self.fstart_step:self.N]
        else: return self.sim
    
    def non_floating_param(self,param:float) -> np.array: 
        N, M = self.N, self.M
        n = self.n_forward_start
        if self.option.payoff.forward_start: 
            spot_forward_start = self.sim[:,self.fstart_step]
            return np.reshape(np.repeat(param*spot_forward_start,n), (M,n)) 
        else: return np.reshape(np.repeat(param,N*M), (M,N))  
    
    def barrier_up(self) -> np.array: 
        return self.non_floating_param(self.option.specification.barrier_up)
    
    def barrier_down(self) -> np.array: 
        return self.non_floating_param(self.option.specification.barrier_down)
    
    def gap(self) -> np.array: 
        return self.non_floating_param(self.option.specification.gap_trigger)  
    
    def rebate(self) -> np.array: 
        return self.non_floating_param(self.option.specification.rebate)     
    
    def binary_amount(self) -> np.array: 
        return self.non_floating_param(self.option.specification.binary_amout)       
    
    def strike(self) -> int: 
        output = np.zeros((self.M, self.N))
        K = self.option.specification.strike
        if self.option.payoff.is_lookback():
            if self.option.payoff.lookback.floating_strike: 
                lb = self.option.payoff.lookback
                ostep = self.option_steps
                lbmc =  MonteCarloLookback(
                    sim=self.sim,
                    forward_start=self.option.payoff.forward_start, 
                    lookback_method=lb.strike_method, 
                    observation_type=lb.strike_observation, 
                    discrete_steps=ostep.strike_lookback_discrete, 
                    begin_window_step=ostep.strike_lookback_window_begin, 
                    end_window_step=ostep.strike_lookback_window_end, 
                    forward_start_step=ostep.forward_start)
                return lbmc.compute()
            else: return self.non_floating_param(K)  
        else: return self.non_floating_param(K)     
    
    def spot(self) -> int: 
        output = np.zeros((self.M, self.N))
        if self.option.payoff.is_lookback():
            if self.option.payoff.lookback.floating_spot: 
                lb = self.option.payoff.lookback
                ostep = self.option_steps
                lbmc =  MonteCarloLookback(
                    sim=self.sim,
                    forward_start=self.option.payoff.forward_start, 
                    lookback_method=lb.spot_method, 
                    observation_type=lb.spot_observation, 
                    discrete_steps=ostep.spot_lookback_discrete, 
                    begin_window_step=ostep.spot_lookback_window_begin, 
                    end_window_step=ostep.spot_lookback_window_end, 
                    forward_start_step=ostep.forward_start)
                return lbmc.compute()
            else: return self.spot_simulation()
        else: return self.spot_simulation() 
    
    def compute_payoff(self): 
        ptool = OptionPayOffTool(
            spot = self.spot(), 
            strike = self.strike(), 
            barrier_up=self.barrier_up(), 
            barrier_down=self.barrier_down(), 
            gap_trigger=self.gap(), 
            rebate=self.rebate(), 
            binary_amount= self.binary_amount(), 
            payoff = self.option.payoff)
        return ptool.payoff_vector() 

    



