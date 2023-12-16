import numpy as np 
from dataclasses import dataclass
from typing import List
from financialmath.instruments.option import (
    Option, 
    LookbackMethod, 
    ObservationType, 
    OptionSteps, 
    ExerciseType)
from financialmath.pricing.schemas import OptionPayOffTool

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
        self.N = self.sim.shape[1]
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
        return np.reshape(a=np.repeat(vec, N), newshape=(M,N))
    
    def compute_lookback_method(self, vectors:np.array) -> np.array: 
        match self.lookback_method: 
            case LookbackMethod.geometric_mean: 
                return np.exp(self.progressive_npfun(np.log(vectors),np.mean))
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
        out[:,e:self.N] = self.vecrep_to_mat(out[:,e-1],self.M,self.N - e)
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
class MonteCarloLeastSquare: 
    simulation : np.array 
    payoff_matrix : np.array 
    dt : float 
    r : float 
    option: Option
    option_steps : OptionSteps
    volatility_matrix : np.array = None 
    spot_lookback_matrix : np.array = None 
    strike_lookback_matrix : np.array = None 
        
    def __post_init__(self):
        self.N = self.option_steps.N
        self.NN = self.payoff_matrix.shape[1] 
        self.bermudan_steps = self.option_steps.bermudan
        self.fstart_step = self.option_steps.forward_start

    @staticmethod
    def coefficients(X, Y) -> np.array: 
        return np.linalg.lstsq(X, Y, rcond=None)[0]
    
    @staticmethod
    def filter_matrix(mat:np.array, i:int, indexes:np.array) -> np.array: 
        try : return mat[indexes, i]
        except Exception as e: pass 
    
    def get_x_matrix(self, i:int, indexes:np.array) -> np.array: 
        n = len(indexes)
        vol = self.filter_matrix(self.volatility_matrix, i, indexes)
        spot = self.filter_matrix(self.simulation, i, indexes)
        spotlb = self.filter_matrix(self.spot_lookback_matrix, i, indexes)
        strikelb = self.filter_matrix(self.strike_lookback_matrix, i, indexes)
        xlist = [np.ones(n)]
        for v in [spot, vol, spotlb, strikelb]: 
            if v is not None: 
                xlist.append(v)
                xlist.append(v**2)
            else: continue
        return np.transpose(np.reshape(np.concatenate(xlist),(len(xlist),n)))
    
    def compute_continuation_payoff(self, discounted_payoff:np.array, 
                                    i:int, indexes:np.array) -> np.array:
        Y, output = discounted_payoff[indexes],np.zeros(len(discounted_payoff))
        X = self.get_x_matrix(i=i, indexes=indexes) 
        coefs = self.coefficients(X=X, Y=Y)
        cpayoff = np.transpose(X.dot(coefs))
        output[indexes] = cpayoff
        return output
    
    def early_exercise_payoff(self, step_range:range) -> np.array: 
        payoff, old_i = self.payoff_matrix[:,self.NN-1], self.NN
        for i in step_range: 
            df = np.exp((old_i-i)*self.dt*-self.r)
            old_i = i 
            discounted_payoff = df*payoff
            actual_payoff = self.payoff_matrix[:,i]
            indexes = np.where(actual_payoff>0)[0]
            continuation_payoff = self.compute_continuation_payoff(
                discounted_payoff=discounted_payoff, 
                i=i, indexes=indexes)
            exercise_indexes = actual_payoff>continuation_payoff
            no_exercise_indexes = np.invert(exercise_indexes)
            payoff[exercise_indexes] = actual_payoff[exercise_indexes]
            payoff[no_exercise_indexes] = discounted_payoff[no_exercise_indexes]
        df = np.exp(-self.r*i*self.dt)
        return df*payoff 
    
    def get_range(self) -> List[int]: 
        match self.option.payoff.exercise: 
            case ExerciseType.american: 
                return list(range(self.NN-2,-1,-1))
            case ExerciseType.bermudan:
                steps = self.bermudan_steps
                diff = self.N - self.NN
                return [s - diff for s in steps]                
            
    def price(self) -> float: 
        payoff = self.early_exercise_payoff(self.get_range())
        price = np.mean(payoff)
        if self.option.payoff.forward_start: 
            df = np.exp(-self.r*self.option_steps.forward_start*self.dt)
            return df*price
        else: return price
    
@dataclass
class MonteCarloPrice: 
    sim : np.array 
    option : Option 
    r : float
    volatility_matrix : np.array = None 

    def __post_init__(self): 
        self.M, self.N = self.sim.shape[0], self.sim.shape[1]
        self.option_steps = self.option.specification.get_steps(self.N)
        self.fstart_step = self.option_steps.forward_start
        self.n_forward_start = self.N - self.fstart_step
        self.t = self.option.specification.tenor.expiry
        self.dt = self.t/self.N
        self.spot_matrix = self.spot()
        self.strike_matrix = self.strike()

    def spot_simulation(self) -> np.array: 
        if self.option.payoff.forward_start: 
            return self.sim[:,self.fstart_step:self.N]
        else: return self.sim
    
    def non_floating_param(self,param:float) -> np.array: 
        N, M = self.N, self.M
        n = self.n_forward_start
        if self.option.payoff.forward_start: 
            spot_forward_start = self.sim[:,self.fstart_step]
            return np.reshape(
                a = np.repeat(param*spot_forward_start,n), 
                newshape = (M,n)) 
        else: return np.reshape(
            a = np.repeat(param,N*M), 
            newshape = (M,N))  
    
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
        K = self.option.specification.strike
        if self.option.payoff.is_lookback():
            if self.option.payoff.lookback.floating_strike: 
                lb = self.option.payoff.lookback
                ostep = self.option_steps
                lbmc =  MonteCarloLookback(
                    sim = self.sim,
                    forward_start = self.option.payoff.forward_start, 
                    lookback_method = lb.strike_method, 
                    observation_type = lb.strike_observation, 
                    discrete_steps = ostep.strike_lookback_discrete, 
                    begin_window_step = ostep.strike_lookback_window_begin, 
                    end_window_step = ostep.strike_lookback_window_end, 
                    forward_start_step = ostep.forward_start
                    )
                return lbmc.compute()
            else: return self.non_floating_param(K)  
        else: return self.non_floating_param(K)     
    
    def spot(self) -> int: 
        if self.option.payoff.is_lookback():
            if self.option.payoff.lookback.floating_spot: 
                lb = self.option.payoff.lookback
                ostep = self.option_steps
                lbmc =  MonteCarloLookback(
                    sim = self.sim,
                    forward_start = self.option.payoff.forward_start, 
                    lookback_method = lb.spot_method, 
                    observation_type = lb.spot_observation, 
                    discrete_steps = ostep.spot_lookback_discrete, 
                    begin_window_step = ostep.spot_lookback_window_begin, 
                    end_window_step = ostep.spot_lookback_window_end, 
                    forward_start_step = ostep.forward_start
                    )
                return lbmc.compute()
            else: return self.spot_simulation()
        else: return self.spot_simulation() 
    
    def is_check_barrier(self, i: int) -> bool: 
        match self.option.payoff.barrier_observation: 
            case ObservationType.in_fine: 
                if i == self.option_steps.N: return True
                else: return False 
            case ObservationType.continuous: return True
            case ObservationType.discrete: 
                if i in self.discrete: return True
                else: return False
            case ObservationType.window: 
                if i<=self.windowe and i>=self.windowb: return True
                else: return False
    
    def check_barrier(self) -> np.array: 
        ptool = OptionPayOffTool(
            spot = self.spot_simulation(), 
            barrier_up = self.barrier_up(), 
            barrier_down = self.barrier_down(), 
            rebate = self.rebate(), 
            strike = np.nan, 
            gap_trigger = np.nan,
            binary_amount = np.nan, 
            payoff = self.option.payoff
            )
        output = ptool.barrier_condition()
        for i in range(1, output.shape[1]): 
            if self.is_check_barrier(i): 
                if self.option.payoff.is_out_barrier():
                    output[:,i] = output[:,i]*output[:,i-1]
                else: output[:,i] = np.minimum(output[:,i] + output[:,i-1], 1)
            else: output[:,i] = output[:,i-1]
        return output 
    
    def compute_payoff_without_barriers(self) -> np.array: 
        ptool = OptionPayOffTool(
            spot = self.spot_matrix, 
            strike = self.strike_matrix, 
            gap_trigger = self.gap(),
            barrier_up = np.nan, 
            barrier_down = np.nan, 
            rebate = np.nan,
            payoff = self.option.payoff, 
            binary_amount = self.binary_amount()
            ) 
        return ptool.payoff_vector_no_barrier()

    def compute_payoff(self) -> np.array: 
        payoff = self.compute_payoff_without_barriers()
        if self.option.payoff.is_barrier():
            barrier_check = self.check_barrier()
            if self.option.payoff.is_out_barrier():
                rebate = self.rebate()
            else: rebate = self.rebate()
            payoff = barrier_check*payoff+rebate
        return payoff
    
    def least_square_object(self) -> MonteCarloLeastSquare: 
        mcls = MonteCarloLeastSquare(
            simulation = self.sim, 
            payoff_matrix = self.compute_payoff(), 
            dt = self.dt, 
            r = self.r, 
            option = self.option, 
            option_steps = self.option_steps, 
            spot_lookback_matrix = None, 
            strike_lookback_matrix = None, 
            volatility_matrix = self.volatility_matrix
            )
        if self.option.payoff.is_lookback():
            if self.option.payoff.lookback.floating_spot: 
                mcls.spot_lookback_matrix=self.spot_matrix
            if self.option.payoff.lookback.floating_strike: 
                mcls.spot_lookback_matrix=self.strike_matrix
        return mcls
    
    def compute_price(self) -> float:       
        match self.option.payoff.exercise: 
            case ExerciseType.european: 
                payoff = self.compute_payoff()
                vector = payoff[:, payoff.shape[1]-1]
                return np.exp(-self.r*self.t)*np.mean(vector)
            case ExerciseType.american: 
                mcls = self.least_square_object()
                return mcls.price() 
            case ExerciseType.bermudan: 
                mcls = self.least_square_object()
                return mcls.price()  

@dataclass
class MonteCarloPricing: 
    initial : float #
    S_up : float #
    r_up : float #
    q_up : float #
    sigma_up : float #
    t_up : float #
    S_down : float #
    sigma_down : float #
    S_up_sigma_up : float #
    S_down_sigma_up : float
    S_up_t_up : float #
    S_down_t_up : float 
    sigma_up_t_up : float #
    sigma_down_t_up : float
    S_uu : float
    S_dd : float
    sigma_uu : float
    sigma_dd : float
    S_uuu : float
    S_ddd : float

    def price(self) -> float: 
        return self.initial
    
    def delta(self, dS:float) -> float: 
        return (self.S_up-self.initial)/dS
    
    def vega(self, dsigma:float) -> float: 
        return (self.sigma_up-self.initial)/dsigma 
    
    def theta(self, dt:float) -> float: 
        return (self.t_up-self.initial)/-dt

    def rho(self, dr:float) -> float: 
        return (self.r_up-self.initial)/dr

    def epsilon(self, dq:float) -> float: 
        return (self.q_up-self.initial)/dq
    
    def gamma(self, dS:float) -> float: 
        return (self.S_up+self.S_down-2*self.initial)/(dS**2)
    
    def volga(self, dsigma:float) -> float: 
        return (self.sigma_up+self.sigma_down-2*self.initial)/(dsigma**2) 
    
    def vanna(self, dS:float, dsigma:float) -> float: 
        delta_up = (self.S_up_sigma_up-self.sigma_up)/dS
        delta = self.delta(dS=dS)
        return (delta_up-delta)/dsigma
    
    def charm(self, dS:float, dt:float) -> float: 
        delta_up = (self.S_up_t_up-self.t_up)/dS
        delta = self.delta(dS=dS)
        return (delta_up-delta)/-dt 

    def veta(self, dt:float, dsigma:float) -> float:  
        vega_up = (self.sigma_up_t_up-self.t_up)/dsigma 
        vega = self.vega(dsigma = dsigma)
        return (vega_up-vega)/-dt
    
    def speed(self, dS: float) -> float: 
        return np.nan
    
    def ultima(self, dsigma:float) -> float: 
        return np.nan

    def color(self,dS: float, dt:float) -> float: 
        return np.nan

    def zomma(self, dS: float, dsigma:float) -> float: 
        return np.nan
    
    def greeks(self, dS: float, dr:float, dq:float, dsigma: float,
               dt : float) -> dict[str,float]: 
        return {'delta':self.delta(dS=dS), 
                'vega':self.vega(dsigma=dsigma), 
                'theta':self.theta(dt=dt), 
                'rho' : self.rho(dr=dr),
                'epsilon':self.epsilon(dq=dq),
                'gamma':self.gamma(dS=dS), 
                'vanna':self.vanna(dS=dS, dsigma=dsigma), 
                'charm':self.charm(dS=dS, dt=dt), 
                'veta':self.veta(dt=dt, dsigma=dsigma), 
                'volga':self.volga(dsigma=dsigma),
                'vera':np.nan, 
                'speed':self.speed(dS=dS), 
                'zomma':self.zomma(dS=dS,dsigma=dsigma), 
                'color':self.color(dS=dS, dt=dt), 
                'ultima':self.ultima(dsigma=dsigma)} 
    



        