from enum import Enum 
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np 
from typing import List
from scipy import interpolate
from financialmath.marketdata.termstructure import ExtrapolatedTotalVarianceTermStructure
from financialmath.tools.tool import MainTool
from financialmath.model.svi import (
    StochasticVolatilityInspired, 
    SSVIFunctions, 
    SurfaceSVI
    )

class VolatilitySurface(ABC): 

    @abstractmethod
    def implied_variance(self, k:np.array, t: np.array) -> np.array: 
        pass 

    @abstractmethod
    def total_variance(self, k:np.array, t: np.array) -> np.array: 
        pass 

    @abstractmethod
    def implied_volatility(self, k:np.array, t: np.array) -> np.array: 
        pass 

    @abstractmethod
    def local_volatility(self, k:np.array, t: np.array) -> np.array: 
        pass 

    @abstractmethod
    def risk_neutral_density(self, k:np.array, t:np.array) -> np.array: 
        pass 

class StrikeType(Enum): 
    strike = 1 
    moneyness = 2 
    log_moneyness = 3
    forward_log_moneyness = 4
    forward_moneyness = 5

@dataclass
class FlatVolatilitySurface(VolatilitySurface): 
    volatility : float 

    def implied_variance(self, k: np.array, t: np.array) -> np.array:
        n = len(k)*len(t)
        return np.repeat(self.volatility**2,n)
    
    def total_variance(self, k: np.array, t: np.array) -> np.array:
        n = len(k)
        return np.repeat(t*(self.volatility**2),n)
    
    def implied_volatility(self, k: np.array, t: np.array) -> np.array:
        n = len(k)*len(t)
        return np.repeat(self.volatility,n)
    
    def local_volatility(self, k: np.array, t: np.array) -> np.array:
        n = len(k)*len(t)
        return np.repeat(self.volatility,n)
    
    def skew(self, k: np.array, t: np.array) -> np.array:
        n = len(k)*len(t)
        return np.repeat(0,n)

@dataclass
class SSVIVolatilitySurface(VolatilitySurface): 

    nu : float 
    rho : float 
    _gamma : float 
    atm_term_structure : ExtrapolatedTotalVarianceTermStructure
    power_law : SSVIFunctions = SSVIFunctions.power_law

    def __post_init__(self): 
        self.ssvi = SurfaceSVI(
            rho=self.rho, 
            nu = self.nu, 
            _gamma=self._gamma, 
            ssvi_function=self.power_law)

    def strike_type(self) -> StrikeType: 
        return StrikeType.log_moneyness
    
    def total_variance(self, t: np.array, k:np.array) -> np.array: 
        atm_tvar = self.atm_term_structure.total_variance(t=t)
        return self.ssvi.total_variance(atm_tvar=atm_tvar, k=k, t=t)
    
    def implied_variance(self, t: np.array, k:np.array) -> np.array: 
        atm_tvar = self.atm_term_structure.total_variance(t=t)
        return self.ssvi.implied_variance(atm_tvar=atm_tvar, k=k, t=t)
    
    def implied_volatility(self, t: np.array, k:np.array) -> np.array: 
        atm_tvar = self.atm_term_structure.total_variance(t=t)
        return self.ssvi.implied_volatility(atm_tvar=atm_tvar, k=k, t=t)
    
    def local_volatility(self, t: np.array, k:np.array) -> np.array: 
        atm_tvar = self.atm_term_structure.total_variance(t=t)
        return self.ssvi.local_volatility(atm_tvar=atm_tvar, k=k, t=t)
    
    def risk_neutral_density(self, t: np.array, k:np.array) -> np.array: 
        atm_tvar = self.atm_term_structure.total_variance(t=t)
        return self.ssvi.risk_neutral_density(atm_tvar=atm_tvar, k=k, t=t)

@dataclass
class SVIObject: 
    indexes : np.array 
    k : np.array 
    t : np.array 
    svi_model : SSVIVolatilitySurface or StochasticVolatilityInspired

    def __post_init__(self): 
        if self.indexes.size == 0: self.compute = False
        else: self.compute = True 

    def total_variance(self): 
        return self.svi_model.total_variance(t=self.t, k=self.k)
    
    def implied_variance(self): 
        return self.svi_model.implied_variance(t=self.t, k=self.k)
    
    def implied_volatility(self): 
        return self.svi_model.implied_volatility(t=self.t, k=self.k)
    
    def local_volatility(self): 
        return self.svi_model.local_volatility(t=self.t, k=self.k)
    
    def risk_neutral_density(self): 
        return self.svi_model.risk_neutral_density(t=self.t, k=self.k)
    
@dataclass
class SVIVolatilitySurface(VolatilitySurface): 
    inputdata : List[StochasticVolatilityInspired]  
    maximum_extrapolated_t : float = 100
    interpolation_method = 'cubic'
    
    def __post_init__(self): 
        self.t_vector = [i.t for i in self.inputdata]
        vt_vector = [i.vt for i in self.inputdata]
        pt_vector = [i.pt for i in self.inputdata]
        ct_vector = [i.ct for i in self.inputdata]
        ut_vector = [i.ut for i in self.inputdata]
        vmt_vector = [i.vmt for i in self.inputdata]
        self.min_obs_t = np.min(self.t_vector)
        self.max_obs_t = np.max(self.t_vector)
        self.svi_min = [i for i in self.inputdata if i.t == self.min_obs_t][0]
        self.svi_max = [i for i in self.inputdata if i.t == self.max_obs_t][0]
        self.atm_term_structure = ExtrapolatedTotalVarianceTermStructure(
            t = self.t_vector, 
            totvariance = vt_vector, 
            max_t = 100
        )
        self.long_term_svi = self.long_term_extrapolation()
        self.short_term_svi = self.short_term_extrapolation()
        self.interpolated_ut = self.interpolate_parameters(ut_vector)
        self.interpolated_ct = self.interpolate_parameters(ct_vector)
        self.interpolated_pt = self.interpolate_parameters(pt_vector)
        self.interpolated_vmt = self.interpolate_parameters(vmt_vector)
    
    def _interpolate_parameters(self, vector:List[float])\
        -> interpolate.interp1d: 
        return interpolate.interp1d(
            x=self.t_vector, 
            y=vector, 
            kind=self.interpolation_method
            )
    
    def _long_term_extrapolation(self) -> SSVIVolatilitySurface: 
        params = self.svi_max.power_law_params()
        return SSVIVolatilitySurface(
            rho = params['rho'], 
            nu = params['nu'], 
            _gamma = params['gamma'], 
            power_law=SSVIFunctions.power_law2, 
            atm_term_structure = self.atm_term_structure
            )
    
    def _short_term_extrapolation(self) -> SSVIVolatilitySurface: 
        params = self.svi_min.power_law_params()
        return SSVIVolatilitySurface(
            rho = params['rho'], 
            nu = params['nu'], 
            _gamma = params['gamma'], 
            power_law=SSVIFunctions.power_law, 
            atm_term_structure = self.atm_term_structure
            ) 
    
    def _in_between_interpolation(self, t:np.array)\
        -> StochasticVolatilityInspired: 
        wt = self.atm_term_structure.total_variance(t=t)
        vt =  wt/t
        pt = self.interpolated_pt(x=t)
        ut = self.interpolated_ut(x=t)
        ct = self.interpolated_ct(x=t)
        vmt = self.interpolated_vmt(x=t)
        return StochasticVolatilityInspired(
            atm_variance = vt, 
            atm_skew = ut, 
            slope_call_wing = ct, 
            slope_put_wing = pt, 
            min_variance = vmt, 
            t = t
        )
    
    def _get_svi_object(self, k: np.array, t: np.array) -> List[SVIObject]:
        i_st = np.where(t<self.min_obs_t)[0]
        i_lt = np.where(t>self.max_obs_t)[0]
        i_bs = np.where((t<=self.max_obs_t) and (t>=self.min_obs_t))[0]
        t_list = [t[i_st], t[i_lt], t[i_bs]]
        k_list = [k[i_st], k[i_lt], k[i_bs]]
        bs_model = self.in_between_interpolation(t=t[i_bs])
        models = [self.short_term_svi, self.long_term_svi, bs_model]
        indexes_list = [i_st, i_lt, i_bs]
        data = zip(indexes_list, k_list, t_list, models)
        return [SVIObject(i,kk,tt,m) for i, kk, tt, m in data]
    
    def _compute_method(self, k: np.array, t: np.array, method:str)\
        -> np.array:
        svi_object = self._get_svi_object(k=k, t=t)
        result = np.zeros(k.shape)
        for s in svi_object: 
            if s.compute: 
                i = s.indexes
                match method: 
                    case 'totalvar': result[i] = s.total_variance()
                    case 'impliedvol': result[i] = s.implied_volatility()
                    case 'localvol': result[i] = s.local_volatility()
                    case 'impliedvar': result[i] = s.implied_variance()
                    case 'rnd': result[i] = s.risk_neutral_density()
            else: continue
        return result
    
    def total_variance(self, k: np.array, t: np.array) -> np.array:
        return self._compute_method(k=k, t=t, method='totalvar')

    def implied_variance(self, k: np.array, t: np.array) -> np.array:
        return self._compute_method(k=k, t=t, method='impliedvar')
    
    def implied_volatility(self, k: np.array, t: np.array) -> np.array:
        return self._compute_method(k=k, t=t, method='impliedvol')
    
    def local_volatility(self, k: np.array, t: np.array) -> np.array:
        return self._compute_method(k=k, t=t, method='localvol')
    
    def risk_neutral_density(self, k: np.array, t: np.array) -> np.array:
        return self._compute_method(k=k, t=t, method='rnd')



        
    
     
        
        
    

