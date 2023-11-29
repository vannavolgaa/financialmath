from dataclasses import dataclass
from scipy import sparse
import numpy as np 
from typing import List
from financialmath.instruments.option import *

@dataclass
class OneFactorGridGenerator: 
    
    final_prices : np.array 
    grid_list : List[sparse.csc_matrix]
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
            tmat =  self.grid_list[n]
            price_vec = tmat.dot(price_vec) 
            price_vec = self.check_path_condition(price_vec,n)
            grid[:, n] = price_vec
        return grid 

@dataclass
class OneFactorFiniteDifferencePricing: 
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
    
    def option_grid(self)-> np.array:
        option_payoff=self.payofftool.payoff_vector()
        grid_shape = (self.M, self.N)
        grid = np.zeros(grid_shape)
        grid[:, self.N-1] = option_payoff
        price_vec = option_payoff
        for n in range(self.N-1, -1, -1):
            tgrid =  self.grid_list[n]
            price_vec = tgrid.dot(price_vec) 
            price_vec = self.check_path_condition(price_vec,n)
            grid[:, n] = price_vec
        return grid 
    
    def vanilla_grid(self, S:float,tmat:List[sparse.csc_matrix], 
                     spec:OptionSpecification) -> np.array:
        option_payoff = self.option.payoff
        payoff = OptionPayoff(
            exercise=option_payoff.exercise, 
            binary=option_payoff.binary, 
            option_type=option_payoff.option_type, 
            gap =option_payoff.gap)
        return self.get_recursive_grid_from_payoff(payoff=payoff,S=S,
                                                   tmat=tmat, spec=spec)
    
    def barrier_out_grid(self, S:float,tmat:List[sparse.csc_matrix], 
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

    def barrier_in_grid(self, S:float,tmat:List[sparse.csc_matrix], 
                        spec:OptionSpecification) -> np.array:
        barrier = self.option.payoff.get_opposite_barrier()
        vanilla = self.vanilla_grid(S=S, tmat=tmat, spec=spec)
        barrier_out = self.barrier_out_grid(S=S, tmat=tmat, 
                                            barrier = barrier, spec=spec)
        return vanilla - barrier_out

    def lookback_grid(self, S:float,tmat:List[sparse.csc_matrix]) -> np.array: 
        return self.grid_fallback()
    
    def option_grid(self, S: float,tmat:List[sparse.csc_matrix],
                    spec:OptionSpecification) -> np.array:
        return self.get_recursive_grid_from_payoff(
            payoff=self.option.payoff,
            S=S,tmat=tmat, spec=spec) 

    def main_grid(self, S:float,tmat:List[sparse.csc_matrix], 
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
    
  