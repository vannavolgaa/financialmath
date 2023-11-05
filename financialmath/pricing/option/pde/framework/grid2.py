import numpy as np
from scipy import sparse, interpolate
from typing import List
from financialmath.instruments.option import * 
from financialmath.pricing.option.schema import OptionValuationFunction
from financialmath.pricing.option.pde.framework.scheme import PDETransitionMatrix
from financialmath.quanttool import QuantTool

@dataclass
class OptionPriceGrids: 
    initial : np.array 
    spot_vector : np.array
    vol_up : np.array = None
    vol_down : np.array = None
    r_up : np.array = None
    q_up : np.array = None
    dt : float = 0
    spot_bump_size: float = 0.01
    volatility_bump_size: float = 0.01
    r_bump_size : float = 0.01
    q_bump_size : float = 0.01

@dataclass
class GridPricingApproximator: 

    grid : OptionPriceGrids 
    S : float 
    interpolation_method: str = 'cubic'

    def __post_init__(self): 
        self.dt = self.grid.dt
        self.spot_vector = self.grid.spot_vector

    def interpolate_value(self, value:float, x: np.array, y:np.array) -> float:
        try: 
            f = interpolate.interp1d(x=x, y=y, kind = self.interpolation_method)
            return f(value).item()
        except: return np.nan 
    
    def option_price(self, S: float, vector: np.array) -> float: 
        return self.interpolate_value(S, self.spot_vector, vector)
    
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
        return math.nan

    def ultima(self) -> float: 
        return np.nan

@dataclass
class PDEPricing(OptionValuationFunction): 

    grid : OptionPriceGrids 
    S : float 
    interpolation_method: str = 'cubic'

    def __post_init__(self): 
        self.spot_vector = self.grid.spot_vector
        self.dt = self.grid.dt
        self.V_0 = self.read_grid(self.grid.initial, 0)
        self.V_dt = self.read_grid(self.grid.initial, 1)
        self.V_0_rp = self.read_grid(self.grid.r_up, 0)
        self.V_0_qp = self.read_grid(self.grid.q_up, 0)
        self.V_0_volu = self.read_grid(self.grid.vol_up, 0)
        self.V_0_vold = self.read_grid(self.grid.vol_down, 0)
        self.V_dt_volu = self.read_grid(self.grid.vol_up, 1)
        self.h_spot = self.grid.spot_bump_size
        self.h_vol = self.grid.volatility_bump_size
        self.h_r = self.grid.r_bump_size
        self.h_q = self.grid.q_bump_size
        self.approximator = GridPricingApproximator(
            self.grid, 
            self.S, 
            self.interpolation_method)
    
    def read_grid(self, grid:np.array, pos: int) -> np.array: 
        try: return grid[:,pos]
        except TypeError: return np.repeat(np.nan, len(self.spot_vector))
    
    def method(self) -> None: 
        return None

    def price(self) -> float: 
        return self.approximator.option_price(self.S, self.V_0) 
    
    def delta(self) -> float:
        return self.approximator.delta(
            S=self.S, vector = self.V_0, 
            h = self.h_spot)
    
    def vega(self) -> float:
        return self.approximator.vega(
            S=self.S,uvec=self.V_0_volu,
            dvec=self.V_0, h=self.h_vol)
    
    def rho(self) -> float:
        return self.approximator.rho(
            S=self.S,uvec=self.V_0_rp,
            dvec=self.V_0, h=self.h_r)
    
    def epsilon(self) -> float:
        return self.approximator.epsilon(
            S=self.S,uvec=self.V_0_qp,
            dvec=self.V_0, h=self.h_q)
    
    def theta(self) -> float:
        return self.approximator.theta(
            S=self.S,uvec=self.V_dt,
            dvec=self.V_0, h=self.dt)
    
    def gamma(self) -> float:
        return self.approximator.gamma(
            S=self.S, vector = self.V_0, 
            h = self.self.h_spot)
    
    def vanna(self) -> float:
        return self.approximator.vanna(
            S = self.S, uvec = self.V_0_volu, dvec = self.V_0, 
            h_S = self.h_spot, h_vol=self.h_vol)
    
    def volga(self) -> float:
        return self.approximator.volga(
            S=self.S, uvec=self.V_0_volu, vec=self.V_0, 
            dvec=self.V_0_vold, h=self.h_vol)
    
    def charm(self) -> float:
        return self.approximator.charm(
            S=self.S, uvec=self.V_dt,
            dvec=self.V_0, h_S=self.h_spot, 
            dt=self.dt)
    
    def veta(self) -> float:
        return self.approximator.veta(
            S=self.S, uvec_dt=self.V_dt_volu, 
            dvec_dt=self.V_dt, 
            uvec=self.V_0_volu, dvec=self.V_0, 
            h_vol=self.h_vol, dt=self.dt)
    
    def speed(self) -> float:
        return self.approximator.speed(
            S=self.S, vector = self.V_0, 
            h = self.h_spot)
    
    def color(self) -> float:
        return self.approximator.color(
            S=self.S, uvec=self.V_dt, dvec=self.V_0, 
            h_S=self.h_spot, dt=self.dt)
    
    def zomma(self) -> float:
        return self.approximator.zomma(
            S=self.S, uvec=self.V_0_volu, 
            dvec=self.V_0, h_S=self.h_spot, 
            h_vol=self.h_vol)
    
    def ultima(self) -> float:
        return self.approximator.ultima()
    
    def vera(self) -> float:
        return self.approximator.vera()

@dataclass
class RecursiveGridGenerator: 
    
    final_prices : np.array 
    transition_matrixes : List[PDETransitionMatrix]
    payoff_object : PayOffObject
    option_steps : OptionSteps
    check_path_cond : bool = True
    
    def __post_init__(self): 
        self.N = len(self.transition_matrixes)
        self.M = len(self.payoff_object.S)
        self.barrier_observation = self.payoff_object.payoff.barrier_obervation
        self.exercise_type = self.payoff_object.payoff.exercise
    
    def early_exercise_condition(self, prices : np.array, n:int) -> np.array: 
        payoff = self.payoff_object.payoff_vector()
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
        condition = self.payoff_object.barrier_condition()
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
            prices = self.early_exercise_condition(prices, n)
            prices = self.touch_barrier_condition(prices, n)
            return prices
        else: return prices
    
    def generate_recursive_grids(self) -> np.array: 
        grid_shape = (self.M, self.N)
        grid = np.zeros(grid_shape)
        grid[:, self.N-1] = self.final_prices
        price_vec = self.final_prices
        for i in range(self.N-1, -1, -1):
            tmatobj =  self.transition_matrixes[i]
            step = tmatobj.step
            tmat = tmatobj.transition_matrix
            price_vec = tmat.dot(price_vec) 
            grid[:, i] = self.check_path_condition(price_vec,step)
        return grid 

@dataclass
class OptionRecursiveGrid: 
    option: Option
    S : float
    dx : float
    M : int 
    transition_matrixes : List[PDETransitionMatrix]

    def __post_init__(self): 
        self.spec = self.option.specification
        self.N = len(self.transition_matrixes)
        self.option_steps = self.spec.get_steps(self.N)
        self.dt = self.spec.tenor.expiry/self.N
    
    def grid_fallback(self): 
        M, N = self.M, self.N
        return np.reshape(np.repeat(np.nan, M*N), (M,N)) 
    
    @staticmethod
    def generate_spot_vector(dx: float, S: float, M : int) -> np.array: 
        spotvec = np.empty(M)
        spotvec[0] = S*np.exp((-dx*M/2))
        for i in range(1,M): 
            spotvec[i] = spotvec[i-1]*np.exp(dx)
        return spotvec
    
    def get_recursive_grid_from_payoff(self, payoff: OptionPayoff, 
                                       S:float, tmat:List[PDETransitionMatrix], 
                                       spec: OptionSpecification)\
                                          -> np.array:
        spot_vector = self.generate_spot_vector(self.dx,S,self.M)
        option_payoff = PayOffObject(spec,payoff,spot_vector)
        grid_gen = RecursiveGridGenerator(
            final_prices=option_payoff.payoff_vector(),
            transition_matrixes=tmat, 
            payoff_object=option_payoff, 
            check_path_cond = True, 
            option_steps=self.option_steps)
        return grid_gen.generate_recursive_grids()
    
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
        print(barrier)
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
                    print(True)
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
        forward_start_price = QuantTool.send_tasks_with_threading(
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





