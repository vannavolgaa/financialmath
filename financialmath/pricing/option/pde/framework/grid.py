import numpy as np
from scipy import sparse, interpolate
from financialmath.instruments.option import Option, OptionSteps
from financialmath.pricing.option.schema import OptionGreeks
from financialmath.instruments.option import * 
from financialmath.pricing.option.payoff import PayOffModule

@dataclass
class OptionRecursiveGrid: 

    option: Option
    transition_matrixes: list[sparse.csc_matrix]
    S : float
    dx : float
    M : int 

    def __post_init__(self): 
        self.N = len(self.transition_matrixes)
        self.option.get_steps(N=self.N)
    
    @staticmethod
    def generate_spot_vector(dx: float, S: float, M : int) -> np.array: 
        spotvec = np.empty(M)
        spotvec[0] = S*np.exp((-dx*M/2))
        for i in range(1,M): 
            spotvec[i] = spotvec[i-1]*np.exp(dx)
        return spotvec

    def get_payoff_module(self, S:float, option : Option) -> np.array: 
        spot_vector = self.generate_spot_vector(S=S, dx=self.dx, M=self.M)
        payoff = option.payoff
        spec = option.specification
        K = spec.strike
        G = spec.gap_trigger
        Bu = spec.barrier_up
        Bd = spec.barrier_down
        Bia = spec.binary_amout
        R = spec.rebate
        if option.payoff.forward_start: 
            return PayOffModule(
                S=spot_vector, K=S*K, Bu=S*Bu, Bd=S*Bd,
                rebate=R, binary_amount=Bia, G = S*G)
        else: 
            return PayOffModule(
                S=spot_vector, K=K, Bu=Bu, Bd=Bd,
                rebate=R, binary_amount=Bia, G = G)  

    def generate_terminal_payoff(self, S: float, option: Option) -> np.array: 
        payoff_module = self.get_payoff_module(S=S, option = option)
        return payoff_module.payoff(option_payoff=option.payoff)

    def generate_barrier_condition(self, S:float, option: Option) -> np.array: 
        payoff_module = self.get_payoff_module(S=S, option=option)
        return payoff_module.barrier_condition(option.payoff.barrier_type)

    def touch_barrier(self, S:float, n:int, option_price: np.array, 
                    option:Option) -> np.array: 
        condition = self.generate_barrier_condition(S=S, option=option)
        match option.payoff.barrier_obervation_type: 
            case ObservationType.continuous: 
                return condition*option_price
            case ObservationType.discrete: 
                if n in option.steps.barrier_discrete: 
                    return condition*option_price
                else: return option_price
            case ObservationType.window:
                end = option.steps.barrier_window_end
                begin = option.steps.barrier_window_begin
                if n >= begin and n <= end: 
                    return condition*option_price
                else: return option_price
            case _: 
                return option_price
    
    def early_exercise(self, S:float, n:int, option_price: np.array, 
                    option:Option) -> np.array:
        terminal_payoff = self.generate_terminal_payoff(S=S, option = option)
        match option.payoff.exercise:
            case ExerciseType.european: 
                return option_price
            case ExerciseType.american:
                return np.maximum(terminal_payoff, option_price)
            case ExerciseType.bermudan: 
                if n in options.steps.bermudan: 
                    return np.maximum(terminal_payoff, option_price)
                else: return option_price
            case _: 
                return np.repeat(np.nan, self.M)

    def option_price_grid(self, S: float, option_price:np.array, 
                        from_step: int, to_step: int, 
                        option: Option=None): 
        n_step = from_step-to_step
        grid_shape = (self.M, n_step)
        grid = np.zeros(grid_shape)
        grid[:, (n_step)-1] = option_price
        price = option_price
        for i in range(n_step-2,-1,-1): 
            n = i + to_step
            tmat = self.transition_matrixes[n]
            price = tmat.dot(price)
            if option is None: 
                grid[:, i] = price
            else: 
                price =  self.early_exercise(S=S, n=n, 
                                            option_price=price, 
                                            option=option)
                price =  self.touch_barrier(S=S, n=n, 
                                            option_price=price, 
                                            option=option)
                grid[:, i] = price  
        return grid  

    def generate_classic_option_grid(self): 
        payoff = self.generate_terminal_payoff(S=self.S, option=self.option)
        return self.option_price_grid(S=self.S, option_price = payoff, 
                                    from_step=self.N,to_step=0, 
                                    option = self.option)


 
@dataclass
class OptionPriceGrids: 
    initial : np.array 
    vol_up : np.array = None
    vol_down : np.array = None
    r_up : np.array = None
    q_up : np.array = None
    spot_bump_size: float = 0.01
    volatility_bump_size: float = 0.01
    r_bump_size : float = 0.01
    q_bump_size : float = 0.01

@dataclass
class PricingGrid: 

    grid : OptionPriceGrids 
    M : int
    S : float 
    dt : float 
    dx: float
    interpolation_method: str = 'cubic'

    def __post_init__(self): 
        self.spot_vector = OptionRecursiveGrid.generate_spot_vector(dx=self.dx, 
                            S=self.S, M=self.M)

    def read_grid(self, grid:np.array, pos: int) -> np.array: 
        try: return grid[:,pos]
        except TypeError: return np.repeat(np.nan, len(self.spot_vector))

    def interpolate_value(self, value:float, x: np.array, y:np.array) -> float:
        try: 
            f = interpolate.interp1d(x=x, y=y, kind = self.interpolation_method)
            return f(value).item()
        except: return np.nan 
    
    def greeks(self) -> OptionGreeks: 

        V_0 = self.read_grid(self.grid.initial, 0)
        V_dt = self.read_grid(self.grid.initial, 1)
        V_0_rp = self.read_grid(self.grid.r_up, 0)
        V_0_qp = self.read_grid(self.grid.q_up, 0)
        V_0_volu = self.read_grid(self.grid.vol_up, 0)
        V_0_vold = self.read_grid(self.grid.vol_down, 0)
        V_dt_volu = self.read_grid(self.grid.vol_up, 1)

        h_spot = self.grid.spot_bump_size
        h_vol = self.grid.volatility_bump_size
        h_r = self.grid.r_bump_size
        h_q = self.grid.q_bump_size

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
        return self.option_price(self.S, V_0)

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


