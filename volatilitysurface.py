from enum import Enum 
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np 
from typing import List
from scipy import interpolate
from financialmath.marketdata.schemas2 import VolatilitySurface
from financialmath.tools.tool import MainTool
from financialmath.model.svi import (
    StochasticVolatilityInspired, 
    SSVIFunctions, 
    SurfaceSVI
    )

class StrikeType(Enum): 
    strike = 1 
    moneyness = 2 
    log_moneyness = 3
    forward_log_moneyness = 4
    forward_moneyness = 5

@dataclass
class ExtrapolatedTotalVarianceTermStructure: 
    t : List[float]
    totvariance : List[float]
    max_t : int = 100

    interpolation_method = 'cubic'

    def __post_init__(self): 
        ordered_dict = MainTool.order_join_lists(
            keys=self.t, 
            values=self.totvariance)
        self.t = np.array(list(ordered_dict.keys()))
        self.totvariance = np.array(list(ordered_dict.values()))
        self.extrapolator = self.extrapolation()

    def extrapolation(self) -> interpolate.interp1d: 
        tvar, t = self.totvariance, self.t
        n = len(self.tvar)
        tvar2, t2, tvar1, t1 = tvar[n-1], t[n-1], tvar[n-2], t[n-2]
        slope = (tvar2-tvar1)/(t2-t1)
        max_tvar = slope*self.max_t
        tvar = np.insert(tvar, [0, len(tvar)], [0, max_tvar])
        t = np.insert(t, [0, len(t)], [0, self.max_t])
        return interpolate.interp1d(t, tvar, kind = self.interpolation_method)
    
    def total_variance(self, t: np.array) -> np.array: 
        return self.extrapolator(x=t)

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
        self.min_obs_t, self.max_obs_t = np.min(t_vector), np.max(t_vector)
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
    
    def interpolate_parameters(self, vector:List[float]) -> interpolate.interp1d: 
        return interpolate.interp1d(
            x=t_vector, 
            y=vector, 
            kind=self.interpolation_method
            )
    
    def long_term_extrapolation(self) -> SSVIVolatilitySurface: 
        params = self.svi_max.power_law_params()
        return SSVIVolatilitySurface(
            rho = params['rho'], 
            nu = params['nu'], 
            _gamma = params['gamma'], 
            power_law=SSVIFunctions.power_law2, 
            atm_term_structure = self.atm_term_structure
            )
    
    def short_term_extrapolation(self) -> SSVIVolatilitySurface: 
        params = self.svi_min.power_law_params()
        return SSVIVolatilitySurface(
            rho = params['rho'], 
            nu = params['nu'], 
            _gamma = params['gamma'], 
            power_law=SSVIFunctions.power_law, 
            atm_term_structure = self.atm_term_structure
            ) 
    
    def in_between_interpolation(self, t:np.array) -> StochasticVolatilityInspired: 
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
    
       
    
        
    
     
        
        
    

