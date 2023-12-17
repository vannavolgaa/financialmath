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

@dataclass
class OneFactorOptionPriceGrids: 
    initial : np.array 
    r_up : np.array = None
    q_up : np.array = None
    sigma_up : np.array = None
    sigma_down : np.array = None
    sigma_uu : np.array = None
    sigma_dd : np.array = None

    interpolation_method = 'cubic'

    def read_grid(self, grid:np.array, pos: int) -> np.array: 
        try: return grid[:,pos]
        except TypeError: return np.repeat(np.nan, len(self.spot_vector))
    
    def __post_init__(self):
        self.vector_0 = self.read_grid(self.initial, 0)
        self.vector_dt = self.read_grid(self.initial, 1)
        self.vector_0_r_up = self.read_grid(self.r_up, 0)
        self.vector_0_q_up = self.read_grid(self.q_up, 0)
        self.vector_0_sigma_up = self.read_grid(self.sigma_up, 0)
        self.vector_0_sigma_down = self.read_grid(self.sigma_down, 0)
        self.vector_dt_sigma_up = self.read_grid(self.sigma_up, 1)
        self.vector_0_sigma_uu = self.read_grid(self.sigma_uu, 0)
        self.vector_0_sigma_dd = self.read_grid(self.sigma_dd, 0)
    
    def interpolate_value(self, value:float, x: np.array, y:np.array) -> float:
        try: 
            f = interpolate.interp1d(
                x = x, 
                y = y, 
                kind = self.interpolation_method
                )
            return f(value).item()
        except: return np.nan 
    
    def option_price(self, S: float, vector: np.array, 
                     spot_vector:np.array) -> float: 
        return self.interpolate_value(S, spot_vector, vector)
    
    def delta(self, S:float, vector: np.array, h: float, 
              spot_vector:np.array) -> float: 
        Su = S*(1+h)
        Sd = S*(1-h)
        Vu = self.option_price(S=Su, vector=vector, spot_vector=spot_vector)
        Vd = self.option_price(S=Sd, vector=vector, spot_vector=spot_vector)
        return (Vu-Vd)/(Su-Sd)

    def gamma(self, S:float, vector: np.array, h: float, 
              spot_vector:np.array) -> float: 
        Su = S*(1+h)
        Sd = S*(1-h)
        delta_up = self.delta(S=Su, vector=vector, h=h, 
                              spot_vector=spot_vector)
        delta_down = self.delta(S=Sd, vector=vector, h=h, 
                                spot_vector=spot_vector)
        return (delta_up-delta_down)/(Su-Sd)

    def speed(self, S:float, vector: np.array, h: float, 
              spot_vector:np.array) -> float: 
        Su = S*(1+h)
        Sd = S*(1-h)
        gamma_up = self.gamma(S=Su, vector=vector, h=h, 
                              spot_vector=spot_vector)
        gamma_down = self.gamma(S=Sd, vector=vector, h=h, 
                                spot_vector=spot_vector)
        return (gamma_up-gamma_down)/(Su-Sd)

    def theta(self, S:float, uvec: np.array, 
            dvec: np.array, h:float, spot_vector:np.array) -> float:  
        Vu = self.option_price(S=S, vector=uvec, spot_vector=spot_vector)
        Vd = self.option_price(S=S, vector=dvec, spot_vector=spot_vector)
        return (Vu-Vd)/h

    def vega(self, S:float, uvec: np.array, 
            dvec: np.array, h:float, spot_vector:np.array) -> float:  
        Vu = self.option_price(S=S, vector=uvec, spot_vector=spot_vector)
        Vd = self.option_price(S=S, vector=dvec, spot_vector=spot_vector)
        return (Vu-Vd)/h
    
    def rho(self, S:float, uvec: np.array, 
            dvec: np.array, h:float, spot_vector:np.array) -> float:  
        Vu = self.option_price(S=S, vector=uvec, spot_vector=spot_vector)
        Vd = self.option_price(S=S, vector=dvec, spot_vector=spot_vector)
        return (Vu-Vd)/h

    def epsilon(self, S:float, uvec: np.array, 
            dvec: np.array, h:float, spot_vector:np.array) -> float:  
        Vu = self.option_price(S=S, vector=uvec, spot_vector=spot_vector)
        Vd = self.option_price(S=S, vector=dvec, spot_vector=spot_vector)
        return (Vu-Vd)/h

    def vanna(self, S:float, uvec: np.array, dvec: np.array, 
            h_S:float, h_vol:float, spot_vector:np.array) -> float: 
        delta_up = self.delta(S=S, vector=uvec, h=h_S, spot_vector=spot_vector)
        delta_down = self.delta(S=S, vector=dvec, h=h_S, 
                                spot_vector=spot_vector)
        return (delta_up-delta_down)/h_vol

    def volga(self, S:float, uvec: np.array, 
            vec: np.array, dvec: np.array, h:float, 
            spot_vector:np.array) -> float: 
        Vu = self.option_price(S=S, vector=uvec, spot_vector=spot_vector)
        V = self.option_price(S=S, vector=vec, spot_vector=spot_vector)
        Vd = self.option_price(S=S, vector=dvec, spot_vector=spot_vector)
        return (Vu+Vd -2*V)/(h**2)

    def charm(self, S:float, uvec: np.array, 
            dvec: np.array, h_S:float, dt:float, 
            spot_vector:np.array) -> float: 
        delta_up = self.delta(S=S, vector=uvec, h=h_S, spot_vector=spot_vector)
        delta_down = self.delta(S=S, vector=dvec, h=h_S, 
                                spot_vector=spot_vector)
        return (delta_up-delta_down)/dt

    def veta(self, S:float, uvec_dt: np.array, dvec_dt: np.array, 
            uvec: np.array, dvec: np.array, h_vol: float, dt:float, 
            spot_vector:np.array) -> float: 
        vega_up = self.vega(S=S, uvec=uvec_dt, dvec=dvec_dt, h=h_vol, 
                            spot_vector=spot_vector) 
        vega_down = self.vega(S=S, uvec=uvec, dvec=dvec, h=h_vol, 
                              spot_vector=spot_vector) 
        return (vega_up-vega_down)/dt

    def zomma(self, S:float, uvec: np.array, dvec: np.array, 
            h_S: float, h_vol: float, spot_vector:np.array) -> float: 
        gamma_up = self.gamma(S=S, vector=uvec, h=h_S, spot_vector=spot_vector)
        gamma_down = self.gamma(S=S, vector=dvec, h=h_S, 
                                spot_vector=spot_vector)
        return (gamma_up-gamma_down)/h_vol
    
    def color(self, S:float, uvec: np.array, dvec: np.array, 
            h_S: float, dt: float, spot_vector:np.array) -> float: 
        gamma_up = self.gamma(S=S, vector=uvec, h=h_S, spot_vector=spot_vector)
        gamma_down = self.gamma(S=S, vector=dvec, h=h_S, 
                                spot_vector=spot_vector)
        return (gamma_up-gamma_down)/dt
    
    def vera(self) -> float: 
        return np.nan

    def ultima(self) -> float: 
        return np.nan
    
    def price(self, S: float, spot_vector: np.array) -> float: 
        return self.option_price(S, self.vector_0, spot_vector)
    
    def greeks(self, S: float, spot_vector: np.array, dS:float, dt:float, 
               dsigma:float, dr:float, dq:float) -> dict[str, float]: 
        return {
            'delta':self.delta(
                S = S,
                vector = self.vector_0,
                h = dS, 
                spot_vector = spot_vector), 
            'vega':self.vega(
                S = S,
                uvec = self.vector_0_sigma_up,
                dvec = self.vector_0,
                h = dsigma, 
                spot_vector = spot_vector), 
            'theta':self.theta(
                S = S,
                uvec = self.vector_dt,
                dvec = self.vector_0,
                h = dt, 
                spot_vector = spot_vector),
            'rho':self.rho(
                S = S,
                uvec = self.vector_0_r_up,
                dvec = self.vector_0,
                h = dr, 
                spot_vector = spot_vector),
            'epsilon':self.epsilon(
                S = S,
                uvec = self.vector_0_q_up,
                dvec = self.vector_0,
                h = dq, 
                spot_vector = spot_vector),
            'gamma':self.gamma(
                S = S,
                vector = self.vector_0,
                h = dS, 
                spot_vector = spot_vector),
            'vanna':self.vanna(
                S = S,
                uvec = self.vector_0_sigma_up,
                dvec = self.vector_0,
                h_S = dS,
                h_vol = dsigma, 
                spot_vector = spot_vector),
            'volga':self.volga(
                S = S,
                uvec = self.vector_0_sigma_up,
                vec = self.vector_0,
                dvec = self.vector_0_sigma_down,
                h = dsigma, 
                spot_vector = spot_vector),
            'charm':self.charm(
                S = S,
                uvec = self.vector_dt,
                dvec = self.vector_0,
                h_S = dS,
                dt = dt, 
                spot_vector = spot_vector),
            'veta':self.veta(
                S = S,
                uvec_dt = self.vector_dt_sigma_up,
                dvec_dt = self.vector_dt,
                uvec = self.vector_0_sigma_up,
                dvec = self.vector_0,
                h_vol = dsigma,
                dt = dt, 
                spot_vector = spot_vector),
            'vera':np.nan,
            'speed':self.speed(
                S = S,
                vector = self.vector_0,
                h = dS, 
                spot_vector = spot_vector),
            'zomma':self.zomma(
                S = S,
                uvec = self.vector_0_sigma_up,
                dvec = self.vector_0,
                h_S = dS,
                h_vol = dsigma, 
                spot_vector = spot_vector),
            'color':self.color(
                S = S,
                uvec = self.vector_dt,
                dvec = self.vector_0,
                h_S = dS,
                dt = dt, 
                spot_vector = spot_vector),
            'ultima':np.nan
            }
    
@dataclass
class OneFactorOptionPriceGridGenerator:
    option : Option 
    matrixes : List[sparse.csc_matrix]
    spot_vector : np.array

    def __post_init__(self): 
        self.spec = self.option.specification
        self.N = len(self.matrixes)
        self.M = len(self.spot_vector)
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
        if ptool.payoff.is_early_exercise():
            prices = self.early_exercise_condition(ptool, prices, n)
        if ptool.payoff.is_barrier():
            prices = self.touch_barrier_condition(ptool,prices, n)
        return prices
    
    def generate_grid(self, ptool: OptionPayOffTool) -> np.array: 
        p0 = ptool.payoff_vector()
        grid_shape = (self.M, self.N)
        grid = np.zeros(grid_shape)
        grid[:, self.N-1] = p0
        price_vec = p0
        for n in range(self.N-1, -1, -1):
            mat =  self.matrixes[n]
            price_vec = mat.dot(price_vec) 
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
