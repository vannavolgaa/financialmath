from dataclasses import dataclass
from scipy import sparse
import numpy as np 
from typing import List
import matplotlib.pyplot as plt 
from financialmath.instruments.option import *


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
class OneFactorGridGenerator: 
    
    final_prices : np.array 
    grid_list : List[PDETransitionMatrix]
    payofftool : PayOffTool
    option_steps : OptionSteps
    check_path_cond : bool = True
    
    def __post_init__(self): 
        self.N = len(self.transition_matrixes)
        self.M = len(self.payoff_object.S)
        self.barrier_observation = self.payoff_object.payoff.barrier_obervation
        self.exercise_type = self.payoff_object.payoff.exercise
    
    def early_exercise_condition(self, prices : np.array, n:int) -> np.array: 
        payoff = self.payofftool.payoff_vector()
        match self.exercise_type:
            case ExerciseType.european: 
                return prices
            case ExerciseType.american:
                return np.maximum(payoff, prices)
            case ExerciseType.bermudan: 
                if n in self.option_steps.bermudan: 
                    return np.maximum(payoff, prices)
                else: return prices
            case _: 
                return np.repeat(np.nan, self.M)

    def touch_barrier_condition(self, prices: np.array, n:int) -> np.array: 
        condition = self.payofftool.barrier_condition()
        match self.barrier_observation: 
            case ObservationType.continuous: 
                return condition*prices
            case ObservationType.discrete: 
                if n in self.option_steps.barrier_discrete: 
                    return condition*prices
                else: return prices
            case ObservationType.window:
                end = self.option_steps.barrier_window_end
                begin = self.option_steps.barrier_window_begin
                if n >= begin and n <= end: 
                    return condition*prices
                else: return prices
            case _: return prices

    def check_path_condition(self, prices : np.array, n:int) -> np.array: 
        if self.check_path_cond: 
            pp = self.touch_barrier_condition(prices, n)
            pp = self.early_exercise_condition(pp, n)
            return pp
        else: return prices
        
    def generate_recursive_grids(self) -> np.array: 
        grid_shape = (self.M, self.N)
        grid = np.zeros(grid_shape)
        grid[:, self.N-1] = self.final_prices
        price_vec = self.final_prices
        for n in range(self.N-1, -1, -1):
            tmatobj =  self.grid_list[n]
            price_vec = tmat.dot(price_vec) 
            price_vec = self.check_path_condition(price_vec,n)
            grid[:, n] = price_vec
        return grid 


@dataclass
class OneFactorPDEPricing: 
    option : Option 
    grid_list : List[sparse.csc_matrix]
    spot_vector : np.array
    S : float 
    
    def __post_init__(self): 
        self.spec = self.option.specification
        self.N = len(self.transition_matrixes)
        self.option_steps = self.spec.get_steps(self.N)
        self.dt = self.spec.tenor.expiry/self.N
        self.payofftool = self.option.payofftool(self.spotvector)
        self.barrier_observation = self.payoff_object.payoff.barrier_obervation
        self.exercise_type = self.payoff_object.payoff.exercise
    
    def early_exercise_condition(self, prices : np.array, n:int) -> np.array: 
        payoff = self.payofftool.payoff_vector()
        match self.exercise_type:
            case ExerciseType.european: 
                return prices
            case ExerciseType.american:
                return np.maximum(payoff, prices)
            case ExerciseType.bermudan: 
                if n in self.option_steps.bermudan: 
                    return np.maximum(payoff, prices)
                else: return prices
            case _: 
                return np.repeat(np.nan, self.M)

    def touch_barrier_condition(self, prices: np.array, n:int) -> np.array: 
        condition = self.payofftool.barrier_condition()
        match self.barrier_observation: 
            case ObservationType.continuous: 
                return condition*prices
            case ObservationType.discrete: 
                if n in self.option_steps.barrier_discrete: 
                    return condition*prices
                else: return prices
            case ObservationType.window:
                end = self.option_steps.barrier_window_end
                begin = self.option_steps.barrier_window_begin
                if n >= begin and n <= end: 
                    return condition*prices
                else: return prices
            case _: return prices

    def check_path_condition(self, prices : np.array, n:int) -> np.array: 
        if self.check_path_cond: 
            pp = self.touch_barrier_condition(prices, n)
            pp = self.early_exercise_condition(pp, n)
            return pp
        else: return prices
        
    def grid_fallback(self): 
        M, N = self.M, self.N
        return np.reshape(np.repeat(np.nan, M*N), (M,N)) 
    
    def get_recursive_grid_from_payoff(self, payoff: OptionPayoff, 
                                       S:float, tmat:List[sparse.csc_matrix], 
                                       spec: OptionSpecification)\
                                          -> np.array:
        option_payoff=PayOffTool(spec,payoff,self.spot_vector).payoff_vector()
        grid_shape = (self.M, self.N)
        grid = np.zeros(grid_shape)
        grid[:, self.N-1] = option_payoff
        price_vec = option_payoff
        for n in range(self.N-1, -1, -1):
            tmatobj =  self.grid_list[n]
            price_vec = tmat.dot(price_vec) 
            price_vec = self.check_path_condition(price_vec,n)
            grid[:, n] = price_vec
        return grid 
    
    def vanilla_grid(self, S:float,tmat:List[PDETransitionMatrix], 
                     spec:OptionSpecification) -> np.array:
        option_payoff = self.option.payoff
        payoff = OptionPayoff(
            exercise=option_payoff.exercise, 
            binary=option_payoff.binary, 
            option_type=option_payoff.option_type, 
            gap =option_payoff.gap)
        return self.get_recursive_grid_from_payoff(payoff=payoff,S=S,
                                                   tmat=tmat, spec=spec)
    
    def barrier_out_grid(self, S:float,tmat:List[PDETransitionMatrix], 
                         barrier:BarrierType, 
                         spec:OptionSpecification) -> np.array: 
        option_payoff = self.option.payoff
        payoff = OptionPayoff(
            exercise=option_payoff.exercise, 
            binary=option_payoff.binary, 
            option_type=option_payoff.option_type, 
            gap =option_payoff.gap, 
            barrier_obervation=option_payoff.barrier_obervation, 
            barrier_type=barrier)
        return self.get_recursive_grid_from_payoff(payoff=payoff,S=S,
                                                   tmat=tmat, spec=spec) 

    def barrier_in_grid(self, S:float,tmat:List[PDETransitionMatrix], 
                        spec:OptionSpecification) -> np.array:
        barrier = self.option.payoff.get_opposite_barrier()
        vanilla = self.vanilla_grid(S=S, tmat=tmat, spec=spec)
        barrier_out = self.barrier_out_grid(S=S, tmat=tmat, 
                                            barrier = barrier, spec=spec)
        return vanilla - barrier_out

    def lookback_grid(self, S:float,tmat:List[PDETransitionMatrix]) -> np.array: 
        return self.grid_fallback()
    
    def option_grid(self, S: float,tmat:List[PDETransitionMatrix],
                    spec:OptionSpecification) -> np.array:
        return self.get_recursive_grid_from_payoff(
            payoff=self.option.payoff,
            S=S,tmat=tmat, spec=spec) 

    def main_grid(self, S:float,tmat:List[PDETransitionMatrix], 
                  spec:OptionSpecification) -> np.array: 
        option_payoff = self.option.payoff
        payoff_type = [option_payoff.is_barrier(),
                       option_payoff.is_lookback()]
        match payoff_type: 
            case [False, False]: 
                return self.option_grid(S=S, tmat=tmat, spec=spec) 
            case [True, False] : 
                if option_payoff.is_in_barrier(): 
                    return self.barrier_in_grid(S=S, tmat=tmat,spec=spec)  
                else: 
                    return self.option_grid(S=S, tmat=tmat, spec=spec) 
            case [True, True]: return self.grid_fallback() 
            case [False, True]: return self.grid_fallback() 
            case _: return self.grid_fallback()
    
    def get_forward_start_price(self, S:float)\
        -> dict[float, float]:
        forward_step = self.option_steps.forward_start
        tmat = [tm for tm in self.transition_matrixes if tm.step>forward_step]
        spec = OptionSpecification(
            strike=self.spec.strike*S, 
            rebate = self.spec.rebate, 
            tenor = self.spec.tenor,
            barrier_down=self.spec.barrier_down*S, 
            barrier_up=self.spec.barrier_up*S, 
            gap_trigger=self.spec.gap_trigger*S, 
            binary_amout=self.spec.binary_amount)
        spot_vector = self.generate_spot_vector(self.dx, S, self.M)
        payoff_object = PayOffObject(spec,self.option.payoff,spot_vector)
        grid = self.get_recursive_grid_from_payoff(payoff= payoff_object, 
            spec=spec, tmat=tmat, S=S)
        gridobj = OptionPriceGrids(initial = grid, spot_vector=spot_vector)
        pricingobj = PDEPricing(grid=gridobj, S=S)
        return {S : pricingobj.price()}

    def forward_start_grid(self) -> np.array: 
        spot_list = list(self.generate_spot_vector(self.dx, self.S, self.M))
        arg_list = [(s,) for s in spot_list]
        forward_start_price = MainTool.send_tasks_with_threading(
            self.get_forward_start_price, 
            arg_list)
        data = dict()
        data = {k: data[k] for k in list(forward_start_price.keys())}
        prices = [data[k] for k in spot_list]
        forward_step = self.option_steps.forward_start
        tmat = [tm for tm in self.transition_matrixes if tm.step<=forward_step]
        grid_gen = RecursiveGridGenerator(
            final_prices=prices,
            transition_matrixes=tmat, 
            payoff_object=self.option.payoff_object(), 
            check_path_cond = False, 
            option_steps=self.option_steps)
        return grid_gen.generate_recursive_grids()

    def generate_grid(self) -> np.array: 
        if self.option.payoff.forward_start: return self.forward_start_grid()
        else: return self.main_grid(self.S, self.transition_matrixes, self.spec)

@dataclass
class MonteCarloPayoff: 

    strike: float 
    barrier_up: float 
    barrier_down: float 
    gap_trigger: float 
    binary_amount : float 
    rebate: float 

    def __post_init__(self): 
        self.K = self.strike
        self.Bu = self.barrier_up
        self.Bd = self.barrier_down
        self.G = self.gap_trigger
        self.binary_amount = self.binary_amout
        self.rebate = self.specification.rebate

    def barrier_condition(self, spot_vector: np.array) -> np.array: 
        S=spot_vector
        b_up = self.Bu
        b_down = self.Bd
        match self.payoff.barrier_type: 
            case BarrierType.up_and_in:
                condition = (S>b_up)
            case BarrierType.up_and_out:
                condition = (S<b_up)
            case BarrierType.down_and_in: 
                condition = (S<b_down)
            case BarrierType.down_and_out: 
                condition = (S>b_down)
            case BarrierType.double_knock_in:
                condition = (S<b_down) & (S>b_up)
            case BarrierType.double_knock_out:
                condition = (S>b_down) & (S<b_up) 
            case _: 
                condition = np.repeat(1, len(S))
        return condition.astype(int) 

    def vanilla_payoff(self, spot_vector: np.array) -> np.array: 
        S = spot_vector
        K = self.K
        match self.payoff.option_type: 
            case OptionalityType.call: 
                return np.maximum(S-K,0)
            case OptionalityType.put: 
                return np.maximum(K-S,0)
            case _: 
                return np.repeat(np.nan, self.M) 

    def gap_payoff(self, spot_vector: np.array) -> np.array: 
        S = spot_vector 
        G = self.G
        K = self.K
        match self.payoff.option_type: 
            case OptionalityType.call: 
                return (S>G).astype(int)*(S-K)
            case OptionalityType.put: 
                return (S<G).astype(int)*(K-S)
            case _: 
                return np.repeat(np.nan, self.M)
    
    def binary_payoff(self, payoff: np.array) -> np.array:
        return (abs(payoff)>0).astype(int)*self.binary_amount*np.sign(payoff)

    def payoff_vector(self, spot_vector:np.array) -> np.array: 
        if self.payoff.gap: 
            payoff = self.gap_payoff(spot_vector=spot_vector)
        else: 
            payoff = self.vanilla_payoff(spot_vector=spot_vector)
        barrier_cond = self.barrier_condition(spot_vector=spot_vector)
        barrier_invcond = np.abs(np.array(barrier_cond) -1)
        payoff = payoff * barrier_cond + barrier_invcond*self.rebate
        if self.payoff.binary: 
            payoff = self.binary_payoff(payoff = payoff)
        return payoff

    def payoff_viewer(self, spot_vector:np.array): 
        payoff = self.payoff_vector(spot_vector=spot_vector)
        plt.plot(self.S, payoff)
        plt.show()

@dataclass
class MonteCarloPricing: 
    option : Option 
    simulation : np.array
    r : float 

    def __post_init__(self): 
        self.t = self.option.specification.tenor.expiry
        self.df = np.exp(-self.r*self.t)
        self.N = self.simulation.shape[1]
        self.option_steps = self.option.specification.get_steps(self.N)



    def strike(self, sim:np.array): 
        pass 

    def breach_barrier(self, sim:np.array)-> np.array: 
        pass







