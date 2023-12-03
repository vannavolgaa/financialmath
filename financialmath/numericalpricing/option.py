import numpy as np 
from dataclasses import dataclass
from scipy import sparse, interpolate
from typing import List
from financialmath.instruments.option import *

@dataclass
class PayOffTool2: 
    spot:np.array or float
    strike: np.array or float
    barier_up: np.array or float
    barrier_down: np.array or float
    gap_trigger: np.array or float
    binary_amount: np.array or float
    rebate: np.array or float
    payoff : OptionPayoff

    def barrier_condition(self) -> np.array or bool: 
        S, b_up, b_down=self.spot, self.b_up, self.b_down
        match self.payoff.barrier_type: 
            case BarrierType.up_and_in:condition = (S>b_up)
            case BarrierType.up_and_out:condition = (S<b_up)
            case BarrierType.down_and_in: condition = (S<b_down)
            case BarrierType.down_and_out: condition = (S>b_down)
            case BarrierType.double_knock_in:condition = (S<b_down)&(S>b_up)
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
class FiniteDifferenceGreeks: 
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

    def price(self): 
        return self.V

    #first order
    def delta(self): 
        return (self.V_S_u-self.V)/self.dS
    
    def vega(self): 
        return (self.V_sigma_u-self.V)/self.dsigma 
    
    def theta(self): 
        return (self.V_t_u-self.V)/self.dt

    def rho(self): 
        return (self.V_r_u-self.V)/self.dt

    def epsilon(self): 
        return (self.V_q_u-self.V)/self.dq
    
    # second order   
    def gamma(self): 
        return (self.V_S_u+self.V_S_d-2*self.V)/(self.S**2) 
    
    def volga(self): 
        return (self.V_sigma_u+self.V_sigma_d-2*self.V)/(self.dsigma**2) 
    
    def vanna(self): 
        delta_up = (self.V_sigma_u_S_u-self.V)/self.dS
        delta_down = (self.V-self.V_sigma_u_S_d)/self.dS
        return (delta_up-delta_down)/self.dsigma
    
    def charm(self): 
        delta_up = (self.V_t_u_S_u-self.V)/self.dS
        delta_down = (self.V-self.V_t_u_S_d)/self.dS
        return (delta_up-delta_down)/self.dt 

    def veta(self): 
        vega_up = (self.V_t_u_sigma_u-self.V)/self.dsigma 
        vega_down = (self.V-self.V_t_u_sigma_d)/self.dsigma 
        return (vega_up-vega_down)/self.dt

    #third order
    def speed(self): 
        delta_up = (self.V_S_u-self.V)/self.dS
        delta_down = (self.V-self.V_S_d)/self.dS
        delta_uu = (self.V_S_uu-self.V_S_u)/self.dS
        delta_dd = (self.V_d - self.V_S_dd)/self.dS
        gamma_up = (delta_uu - delta_up)/self.dS
        gamma_down = (delta_down - delta_dd)/self.dS
        return (gamma_up-gamma_down)/self.dS
    
    def ultima(self): 
        vega_up = (self.V_sigma_u-self.V)/self.dsigma
        vega_down = (self.V-self.V_sigma_d)/self.dsigma
        vega_uu = (self.V_sigma_uu-self.V_sigma_u)/self.dsigma
        vega_dd = (self.V_sigma_d - self.V_sigma_dd)/self.dsigma
        volga_up = (vega_uu - vega_up)/self.dsigma
        volga_down = (vega_down - vega_dd)/self.dsigma
        return (volga_up-volga_down)/self.dsigma

    def color(self): 
        d = (self.dt*(self.dS)**2)
        return (self.V_t_u_S_u+self.V_t_u_S_u-2*self.V_t_u)/d

    def zomma(self): 
        d = (self.dsigma*(self.dS)**2)
        return (self.V_sigma_u_S_u+self.V_sigma_u_S_d-2*self.V_sigma_u)/d

@dataclass
class GridOptionPrices: 
    V : float 
    V_S_u : float 
    V_S_d : float 
    V_S_uu : float 
    V_S_dd : float
    V_t_u : float 
    V_S_u_t_u : float 
    V_S_d_t_u : float 

@dataclass
class SpotFactorFiniteDifferencePricing: 
    option : Option 
    grid_list : List[sparse.csc_matrix]
    spot_vector : np.array
    S : float 
    dt : float 
    dS : float 
    interpolation_method: str = 'cubic'
    
    def __post_init__(self): 
        self.spec = self.option.specification
        self.N = len(self.transition_matrixes)
        self.option_steps = self.spec.get_steps(self.N)
        self.dt = self.spec.tenor.expiry/self.N
        self.ptool = PayOffTool2(
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

    def early_exercise_condition(self, ptool: PayOffTool2,
                                  prices : np.array, n:int) -> np.array: 
        payoff = ptool.payoff_vector()
        match self.option.payoff.exercise_type:
            case ExerciseType.european: return prices
            case ExerciseType.american: return np.maximum(payoff, prices)
            case ExerciseType.bermudan: 
                if n in self.option_steps.bermudan: 
                    return np.maximum(payoff, prices)
                else: return prices
    
    def touch_barrier_condition(self, ptool: PayOffTool2,
                                prices: np.array, n:int) -> np.array: 
        condition = ptool.barrier_condition()
        match self.option.payoff.barrier_observation: 
            case ObservationType.continuous: return condition*prices
            case ObservationType.in_fine: return prices
            case ObservationType.discrete: 
                if n in self.option_steps.barrier_discrete: 
                    return condition*prices
                else: return prices
            case ObservationType.window:
                end = self.option_steps.barrier_window_end
                begin = self.option_steps.barrier_window_begin
                if n >= begin and n <= end: return condition*prices
                else: return prices
            case _: return prices
            
    def check_path_condition(self, ptool: PayOffTool2, 
                             prices : np.array, n:int) -> np.array: 
        pp = self.touch_barrier_condition(ptool,prices, n)
        pp = self.early_exercise_condition(ptool, pp, n)
        return pp
    
    def generate_grid(self, ptool: PayOffTool2) -> np.array: 
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
            barrier_obervation=option_payoff.barrier_obervation, 
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

    def interpolate_price_from_grid(self) -> GridOptionPrices: 
        grid = self.option_price_grid()
        grid0, griddt = grid[:,0], grid[:,1]
        S, dS, dt = self.S, self.dS, self.dt 
        Su, Sd, Suu, Sdd = S*(1+dS), S*(1-dS), S*(1+2*dS), S*(1-2*dS)
        p  = [self.interpolate_value(s, self.spot_vector,grid0) 
            for s in [S,Su,Sd,Suu,Sdd]]
        pdt  = [self.interpolate_value(s, self.spot_vector,griddt) 
                for s in [S,Su,Sd]]
        return GridOptionPrices(
            V = p[0], V_S_d=p[2], V_S_u=p[1], V_S_dd=p[4], V_S_uu=p[3], 
            V_S_d_t_u=pdt[2], V_S_u_t_u=pdt[1], V_t_u=pdt[0])
        
@dataclass
class SpotSimulationPricing:
    pass