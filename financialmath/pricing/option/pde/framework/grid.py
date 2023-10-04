import numpy as np
from scipy import sparse, interpolate
from financialmath.instruments.option import Option
from financialmath.pricing.option.obj import OptionGreeks
from financialmath.instruments.option import * 

@dataclass
class OptionSteps:
    tenor : OptionTenor 
    N : int 

    @staticmethod
    def t_to_step(t, dt): 
        factor = t/dt
        if not np.isnan(factor):
            return round(t/dt)
        else: return np.nan

    def __post_init__(self): 
        tenor = self.tenor
        N = self.N 
        dt = tenor.expiry/N 
        self.expiry = N
        self.bermudan = [self.t_to_step(t=t, dt=dt) for t in tenor.bermudan]
        self.barrier_discrete = [self.t_to_step(t=t, dt=dt) 
                                for t in tenor.barrier_discrete]
        self.barrier_window_end  = self.t_to_step(t=tenor.barrier_window_end, dt=dt)
        self.barrier_window_begin = self.t_to_step(t=tenor.barrier_window_begin, dt=dt) 
        self.lookback_discrete = [self.t_to_step(t=t, dt=dt) 
                                for t in tenor.lookback_discrete]
        self.lookback_window_begin  = self.t_to_step(t=tenor.lookback_window_begin, dt=dt) 
        self.lookback_window_end  = self.t_to_step(t=tenor.lookback_window_end, dt=dt)
        self.choser = self.t_to_step(t=tenor.choser, dt=dt)

@dataclass
class RecursiveGrid: 

    instrument: Option
    transition_matrixes: list[sparse.csc_matrix]
    spot_vector : np.array

    def __post_init__(self): 
        self.N = len(self.transition_matrixes)
        self.M = len(self.spot_vector)
        self.steps = OptionSteps(tenor=self.instrument.specification.tenor, N=self.N)

    def binary_payoff(self, option_payoff: np.array) -> np.array:
        if self.instrument.exotic.binary:
            payoff = self.instrument.exotic.binary_payoff
            return (option_payoff>0).astype(int)*payoff
        else: 
            return option_payoff
    
    def barrier_payoff_condition(self) -> np.array: 
        b_up = self.instrument.exotic.barrier.up
        b_down = self.instrument.exotic.barrier.down
        match self.instrument.exotic.barrier.barrier_type: 
            case BarrierType.up_and_in:
                condition = (self.spot_vector>b_up)
            case BarrierType.up_and_out:
                condition = (self.spot_vector<b_up)
            case BarrierType.down_and_in: 
                condition = (self.spot_vector<b_down)
            case BarrierType.down_and_out: 
                condition = (self.spot_vector>b_down)
            case BarrierType.double_knock_in:
                condition = (self.spot_vector<b_down) and (self.spot_vector>b_up)
            case BarrierType.double_knock_in:
                condition = (self.spot_vector>b_down) and (self.spot_vector<b_up) 
            case _: 
                condition = np.repeat(1, self.M)
        return condition.astype(int)

    def barrier_payoff(self, option_payoff: np.array) -> np.array: 
        try:
            condition = self.barrier_payoff_condition()
            return condition*option_payoff
        except AttributeError: 
            return option_payoff
    
    def payoff(self) -> np.array: 
        K = self.instrument.specification.strike
        S = self.spot_vector
        match self.instrument.option:
            case OptionalityType.call: 
                payoff = np.maximum(S-K, 0)
            case OptionalityType.put: 
                payoff = np.maximum(K-S, 0)
            case _: 
                return np.repeat(np.nan, self.M)
        payoff = self.barrier_payoff(option_payoff=payoff)
        return self.binary_payoff(option_payoff=payoff)
                
    def early_exercise(self, option_price: np.array, n: int) -> np.array: 
        match self.instrument.exercise:
            case ExerciseType.european: 
                return option_price
            case ExerciseType.american:
                return np.maximum(self.payoff(), option_price)
            case ExerciseType.bermudan: 
                if n in self.steps.bermudan: 
                    return np.maximum(self.payoff(), option_price)
                else: return option_price
            case _: 
                return np.repeat(np.nan, self.M)
    
    def touch_barrier(self, option_price: np.array, n: int) -> np.array: 
        condition = self.barrier_payoff_condition()
        match self.instrument.barrier.observation: 
            case ObservationType.continuous: 
                return condition*option_price
            case ObservationType.discrete: 
                if n in self.steps.barrier_discrete: 
                    return condition*option_price
                else: return option_price
            case ObservationType.window:
                end = self.steps.barrier_window_end
                begin = self.steps.barrier_window_begin
                if n >= begin and n <= end: 
                    return condition*option_price
                else: return option_price

    def pathbound(self, option_price: np.array, n: int) -> np.array: 
        try : price = self.touch_barrier(option_price=option_price, n=n)
        except AttributeError: 
            price = option_price
        return self.early_exercise(option_price=price, n=n)

    def generate(self) -> np.array: 
        grid_shape = (self.M,self.N)
        grid = np.zeros(grid_shape)
        price = self.payoff()
        grid[:, self.N-1] = price
        for i in range(self.N-2, -1, -1): 
            tmat = self.transition_matrixes[i+1]
            price = tmat.dot(price)
            price = self.pathbound(option_price=price, n=i)
            grid[:,i] = price
        return grid

@dataclass
class OptionRecursiveGrid: 

    option: Option
    transition_matrixes: list[sparse.csc_matrix]
    spot_vector : np.array

    def __post_init__(self): 
        self.N = len(self.transition_matrixes)
        self.M = len(self.spot_vector)
        self.specification = self.option.specification
        self.payoff = self.option.payoff
        self.steps = OptionSteps(tenor=self.instrument.specification.tenor, N=self.N)
    
    def gap_condition(self, G: float) -> np.array: 
        S = self.spot_vector
        match self.payoff.option:
            case OptionalityType.call: 
                return (S>G).astype(int)
            case OptionalityType.put: 
                return (S<G).astype(int)
            case _: 
                return np.repeat(np.nan, self.M)

    def terminal_payoff(
        self, K:float, b_up: float, b_down: float, 
        binary_payoff: float, G: float
        ) -> np.array: 

        S = self.spot_vector
        gap_cond = self.gap_condition(G=G)

        match self.payoff.option:
            case OptionalityType.call: 
                if self.payoff.option.gap: payoff = gap_cond*(S - K)
                else: payoff = np.maximum(S-K, 0)
            case OptionalityType.put: 
                if self.payoff.option.gap: payoff = gap_cond*(K - S)
                else: payoff = np.maximum(K-S, 0)
            case _: 
                return np.repeat(np.nan, self.M)

        barrier_cond = barrier_payoff_condition(b_up=b_up, d_down=b_down)
        payoff = payoff * barrier_cond
        if self.payoff.binary: 
            binary_cond = (payoff > 0).astype(int)
            return binary_cond * binary_payoff
        else: return payoff
                 
    def barrier_payoff_condition(
        self, b_up: float, b_down:float
        ) -> np.array: 
        
        S=self.spot_vector
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
                condition = (S<b_down) and (S>b_up)
            case BarrierType.double_knock_in:
                condition = (S>b_down) and (S<b_up) 
            case _: 
                condition = np.repeat(1, self.M)
        return condition.astype(int)

    def path_boundaries(
        self, option_price: np.array,K:float, 
        b_up: float, b_down: float, n : int,
        binary_payoff: float, G : float) -> np.array: 
        price = self.path_touch_barrier(option_price=option_price, n=n,
                                        b_up=b_up, b_down=b_down)
        price = self.early_exercise(option_price=price, n=n,
                                    K=K, b_up=b_up, b_down=b_down, 
                                    binary_payoff=binary_payoff, G=G)
        return price 

    def path_touch_barrier(
        self, option_price: np.array, n: int, 
        b_up : float, b_down:float
        ) -> np.array: 
        condition = self.barrier_payoff_condition(b_up=b_up, b_down=b_down)
        match self.instrument.barrier.observation: 
            case ObservationType.continuous: 
                return condition*option_price
            case ObservationType.discrete: 
                if n in self.steps.barrier_discrete: 
                    return condition*option_price
                else: return option_price
            case ObservationType.window:
                end = self.steps.barrier_window_end
                begin = self.steps.barrier_window_begin
                if n >= begin and n <= end: 
                    return condition*option_price
                else: return option_price
            case _: 
                return option_price
    
    def early_exercise(
        self, option_price: np.array,K:float, 
        b_up: float, b_down: float, n:int,
        binary_payoff: float, G : float
        ) -> np.array:

        terminal_payoff = self.terminal_payoff(K=K, b_up=b_up, 
                                            b_down=b_down, 
                                            binary_payoff=binary_payoff, 
                                            G = G)
        match self.payoff.exercise:
            case ExerciseType.european: 
                return option_price
            case ExerciseType.american:
                return np.maximum(terminal_payoff, option_price)
            case ExerciseType.bermudan: 
                if n in self.steps.bermudan: 
                    return np.maximum(terminal_payoff, option_price)
                else: return option_price
            case _: 
                return np.repeat(np.nan, self.M)
    
@dataclass
class BumpGrid: 
    volatility_up : bool = True
    volatility_down : bool = True
    r_up : bool = True
    q_up : bool = True
    spot_bump_size: float = 0.01
    volatility_bump_size: float = 0.01
    r_bump_size : float = 0.01
    q_bump_size : float = 0.01

@dataclass
class GridObject: 
    initial : np.array 
    bump : BumpGrid = BumpGrid()
    vol_up : np.array = None
    vol_down : np.array = None
    r_up : np.array = None
    q_up : np.array = None

@dataclass
class PricingGrid: 

    S : float 
    dt : float 
    spot_vector : np.array
    grid : GridObject 
    interpolation_method = 'cubic'

    def read_grid(self, grid:np.array, pos: int) -> np.array: 
        try: return grid[:,pos]
        except TypeError: return np.repeat(np.nan, len(self.spot_vector))

    def interpolate_value(self, value:float, x: np.array, y:np.array) -> float:
        f = interpolate.interp1d(x=x, y=y, kind = self.interpolation_method)
        return f(value)
    
    def greeks(self) -> OptionGreeks: 

        V_0 = self.read_grid(self.grid.initial, 0)
        V_dt = self.read_grid(self.grid.initial, 1)
        V_0_rp = self.read_grid(self.grid.r_up, 0)
        V_0_qp = self.read_grid(self.grid.q_up, 0)
        V_0_volu = self.read_grid(self.grid.vol_up, 0)
        V_0_vold = self.read_grid(self.grid.vol_down, 0)
        V_dt_volu = self.read_grid(self.grid.vol_up, 1)

        h_spot = self.grid.bump.spot_bump_size
        h_vol = self.grid.bump.volatility_bump_size
        h_r = self.grid.bump.r_bump_size
        h_q = self.grid.bump.q_bump_size

        delta = self.delta(S=self.S, vector = V_0, h = h_spot)
        vega = self.vega(S=self.S,uvec=V_0_volu,dvec=V_0, h=h_vol)
        rho = self.rho(S=self.S,uvec=V_0_rp,dvec=V_0, h=h_r)
        epsilon = self.epsilon(S=self.S,uvec=V_0_qp,dvec=V_0, h=h_q)
        theta = self.theta(S=self.S,uvec=V_dt,dvec=V_0, h=self.dt)
        gamma = self.gamma(S=self.S, vector = V_0, h = h_spot)
        vanna = self.vanna(S = self.S, uvec = V_0_volu, dvec = V_0, 
                        h_S = h_spot, h_vol=h_vol)
        volga = self.volga(S=self.S, uvec=V_0_volu, vec=V_0, dvec=V_0_vold, 
                        h=h_vol)
        charm = self.charm(S=self.S, uvec=V_dt,dvec=V_0, h_S=h_spot, 
                            dt=self.dt)
        veta = self.veta(S=self.S, uvec_dt=V_dt_volu, dvec_dt=V_dt, 
                        uvec=V_0_volu, dvec=V_0, h_vol=h_vol, dt=self.dt)
        speed = self.speed(S=self.S, vector = V_0, h = h_spot)
        color = self.color(S=self.S, uvec=V_dt, dvec=V_0, 
                        h_S=h_spot, dt=self.dt)
        zomma = self.zomma(S=self.S, uvec=V_0_volu, dvec=V_0, 
                        h_S=h_spot, h_vol=h_vol)
        ultima = self.ultima()
        vera = self.vera()

        return OptionGreeks(delta=delta, vega=vega, 
                            theta=theta, rho=rho, 
                            epsilon=epsilon, gamma=gamma, 
                            vanna=vanna, volga=volga, 
                            charm=charm, veta=veta, 
                            vera=np.nan, speed=speed, 
                            zomma=zomma, color=color, 
                            ultima=ultima)
    
    def price(self) -> float: 
        V_0 = self.read_grid(self.grid.initial, 0)
        return self.option_price(self.S, vector)

    def option_price(self, S: float, vector) -> float: 
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

