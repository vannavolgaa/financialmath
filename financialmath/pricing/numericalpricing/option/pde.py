from dataclasses import dataclass
from typing import NamedTuple, List
import numpy as np 
from scipy import sparse, interpolate
from financialmath.instruments.option import (
    Option, 
    BarrierType,  
    ExerciseType,
    OptionPayoff, 
    ObservationType)
from financialmath.pricing.schemas import OptionPayOffTool

class OneFactorOptionPriceGrids(NamedTuple): 
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

class OneFactorFiniteDifferenceGreeks: 

    def __init__(self, inputdata: OneFactorOptionPriceGrids, 
                      interpolation_method: str = 'cubic'): 
        self.S = inputdata.S
        self.dt = inputdata.dt
        self.dS = inputdata.dS
        self.dsigma = inputdata.dsigma
        self.dr = inputdata.dr
        self.dq = inputdata.dq
        self.spot_vector = inputdata.spot_vector
        self.V_0 = self.read_grid(inputdata.initial, 0)
        self.V_dt = self.read_grid(inputdata.initial, 1)
        self.V_0_rp = self.read_grid(inputdata.r_up, 0)
        self.V_0_qp = self.read_grid(inputdata.q_up, 0)
        self.V_0_volu = self.read_grid(inputdata.vol_up, 0)
        self.V_0_vold = self.read_grid(inputdata.vol_down, 0)
        self.V_dt_volu = self.read_grid(inputdata.vol_up, 1)
        self.V_0_voluu = self.read_grid(inputdata.vol_uu, 0)
        self.V_0_voldd = self.read_grid(inputdata.vol_dd, 0)
        self.interpolation_method = interpolation_method
        
    def read_grid(self, grid:np.array, pos: int) -> np.array: 
        try: return grid[:,pos]
        except TypeError: return np.repeat(np.nan, len(self.spot_vector))

    def interpolate_value(self, value:float, x: np.array, y:np.array) -> float:
        try: 
            f = interpolate.interp1d(
                x = x, 
                y = y, 
                kind = self.interpolation_method
                )
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
            'ultima':np.nan
            }
        
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
            barrier_up=self.spec.barrier_up, 
            rebate=self.spec.rebate, 
            gap_trigger=self.spec.gap_trigger,
            binary_amount=self.spec.binary_amout,
            payoff = self.option.payoff
            )

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
            exercise = option_payoff.exercise, 
            binary = option_payoff.binary, 
            option_type = option_payoff.option_type, 
            gap = option_payoff.gap
            )
        ptool = self.ptool
        ptool.payoff = payoff
        return self.generate_grid(ptool)
    
    def barrier_out_grid(self, barrier: BarrierType) -> np.array: 
        option_payoff = self.option.payoff
        payoff = OptionPayoff(
            exercise = option_payoff.exercise, 
            binary = option_payoff.binary, 
            option_type = option_payoff.option_type, 
            gap = option_payoff.gap, 
            barrier_observation = option_payoff.barrier_observation, 
            barrier_type = barrier
            )
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
