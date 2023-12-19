import numpy as np
from dataclasses import dataclass
from typing import List
from enum import Enum

@dataclass
class JWSVITool:
    atm_variance : float or np.array 
    atm_skew : float or np.array 
    slope_put_wing : float or np.array 
    slope_call_wing : float or np.array 
    min_variance : float or np.array
    t : float or np.array 

    def __post_init__(self): 
        self.vt = self.atm_variance
        self.ut = self.atm_skew
        self.pt = self.slope_put_wing
        self.ct = self.slope_call_wing
        self.vmt = self.min_variance
        self.wt = self.vt*self.t
        self.b = self.get_b()
        self.p = self.get_p()
        self.beta = self.get_beta()
        self.alpha = self.get_alpha()
        self.m = self.get_m()
        self.a_and_s = self.get_s_a()
        self.a = self.a_and_s['a']
        self.s = self.a_and_s['s']
    
    def get_fallback(self) -> float or np.array: 
        n = np.max([len(l) for l in [self.vt, self.ut, self.pt, 
                                     self.ct, self.vmt, self.t]])
        if n == 1: return np.nan
        else: return np.repeat(np.nan, n)
    
    def get_b(self) -> float or np.array: 
        try: 
            wt = self.vt*self.t
            sqwt = np.sqrt(wt)
            ctpt = self.ct+self.pt
            return .5*sqwt*ctpt 
        except Exception as e: return self.get_fallback()
    
    def get_p(self) -> float or np.array: 
        try: 
            ctpt = self.ct+self.pt
            return 1 - 2 * self.pt/ctpt
        except Exception as e: return self.get_fallback()
    
    def get_beta(self) -> float or np.array: 
        p = self.p
        try:
            ctpt = self.ct+self.pt
            return p - 4*self.ut/ctpt
        except Exception as e: return self.get_fallback()
    
    def get_alpha(self) -> float or np.array: 
        beta = self.beta
        try: return np.sign(beta)*np.sqrt(-1+1/(beta**2))
        except Exception as e: return self.get_fallback()
    
    def get_m(self) -> float or np.array: 
        a, p, b = self.alpha, self.p, self.b
        try : 
            factor_num = self.vt - self.vmt 
            factor_den = -p + np.sign(a) * np.sqrt(1+a**2) - a*np.sqrt(1-p**2)
            factor = factor_num/factor_den
            return factor*self.t/b
        except Exception as e: return self.get_fallback()
    
    def is_m_equal_zero(self) -> bool or List[bool]: 
        m = self.m
        if not isinstance(m, float): 
            return [m == 0 for m in list(m)] 
        else: return (self.m == 0)
    
    def get_s_a(self) -> dict[str, float or np.array]: 
        alpha, m, b, p = self.alpha, self.m, self.b, self.p
        try: 
            sq1p =  np.sqrt(1-p**2)
            s = alpha*m
            a = self.vmt*self.t - b*s*sq1p
            cond = self.is_m_equal_zero()
            a2 = self.t*(self.vmt + self.vt*sq1p)/(1-sq1p)
            s2 = (self.vt*self.t - a2)/b 
            if not isinstance(cond, list): 
                if cond: a, s = a2, s2
            else: 
                condpos = np.where(np.array(cond) == True)
                a[condpos] = a2[condpos]
                s[condpos] = s2[condpos]
            return {'a' : a, 's': s}
        except Exception as e: return {
            'a':self.get_fallback(), 
            's':self.get_fallback()
            }

    def dwdk(self, k: float or np.array) -> float or np.array: 
        b, p = self.b, self.p 
        kminusm = (k-self.m)
        factor = kminusm/np.sqrt(kminusm**2 + self.s**2)
        return b*(p+factor)
    
    def d2wd2k(self, k: float or np.array) -> float or np.array: 
        u= (k-self.m)
        v = np.sqrt(u**2 + self.s**2)
        vprime = u/v 
        return self.b*(v - vprime*u)/(u**2 + self.s**2)
    
    def dbdt(self) -> float or np.array: 
        return np.sqrt(self.vt)*self.ctpt/(4*np.sqrt(self.t))
    
    def dmdt(self) -> float or np.array: 
        a = self.alpha
        p = self.p
        factor_num = self.vt - self.vmt 
        factor_den = -p + np.sign(a) * np.sqrt(1+a**2) - a*np.sqrt(1-p**2)
        factor = factor_num/factor_den
        return factor*(self.b-self.t*self.dbdt())/(self.b**2)
    
    def dsdt(self) -> float or np.array:
        dsdt = self.alpha*self.dmdt()
        cond = self.is_m_equal_zero()
        num1 = (self.vt-self.a/self.t)*self.b
        num2 = (self.vt*self.t-self.a)*self.dbdt()
        dsdt2 =  (num1 + num2)/(self.b**2)
        if not isinstance(cond, list): 
            if cond:dsdt = dsdt2  
        else: 
            condpos = np.where(np.array(cond) == True)
            dsdt[condpos] = dsdt2[condpos]
        return dsdt

    def dadt(self) -> float or np.array:
        sq1p =  np.sqrt(1-self.p**2)
        dadt = self.vmt - sq1p*(self.dbdt()*self.s + self.dsdt()*self.b)
        cond = self.is_m_equal_zero()
        dadt2 = self.a/self.t 
        if not isinstance(cond, list): 
            if cond:dadt = dadt2  
        else: 
            condpos = np.where(np.array(cond) == True)
            dadt[condpos] = dadt2[condpos]
        return dadt
    
    def dfdt(self, k: float or np.array) -> float or np.array:
        return 2 * (self.dsdt()*self.s - self.dmdt()*(k-self.m))
    
    def dsqfdt(self, k: float or np.array) -> float or np.array:
        den = 2 * np.sqrt((k-self.m)**2 + self.s**2)
        return self.dfdt(k=k)/den
    
    def functiong(self, k: float or np.array) -> float or np.array:
        return self.p*(k-self.m) + np.sqrt((k-self.m)**2 + self.s**2)
    
    def dgdt(self, k: float or np.array) -> float or np.array:
        return -self.p*self.dmdt() + self.dsqfdt(k=k)

    def dwdt(self, k: float or np.array) -> float or np.array: 
        b=self.b
        return self.dadt()+b*self.dgdt(k=k)+self.dbdt()*self.functiong(k=k)

class SSVIFunctions(Enum):
    heston_like = 1 
    power_law = 2
    power_law2 = 3

@dataclass
class StochasticVolatilityInspired:
    atm_variance : float or np.array 
    atm_skew : float or np.array 
    slope_put_wing : float or np.array 
    slope_call_wing : float or np.array 
    min_variance : float or np.array
    t : float or np.array 

    def __post_init__(self): 
        self.tool = JWSVITool(
            atm_variance=self.atm_variance, 
            atm_skew=self.atm_skew, 
            slope_call_wing=self.slope_call_wing, 
            slope_put_wing=self.slope_put_wing, 
            min_variance=self.min_variance, 
            t = self.t
            )
        self.a = self.tool.a
        self.b = self.tool.b
        self.m = self.tool.m 
        self.s = self.tool.s 
        self.p = self.tool.p

    def parameters_check(self) -> int: 
        cond_rho = (np.abs(self.p) < 1)
        cond_b = (self.b >= 0)
        cond_s = (self.s > 0)
        cond_f = (self.a + self.b*self.s*np.sqrt(1-self.p**2)>0)
        if cond_rho and cond_b and cond_s and cond_f: return 0
        else: return 1
    
    def butterfly_check(self) -> int: 
        sqwt = np.sqrt(self.vt*self.t)
        mcptct = np.maximum(self.ct, self.pt)
        ptct = self.pt+self.ct
        cond1 = (sqwt*mcptct<2)
        cond2 = (ptct*mcptct<=2)
        if cond1 and cond2: return 0 
        else: return 1

    def calendar_spread_check(self, f_t: np.array, p_t : np.array, 
                              atmtvar_t : np.array, 
                              ascending:bool = True) -> int: 
        p = np.sqrt(1-self.min_variance/self.atm_variance)
        pf = 2*self.atm_skew/np.sqrt(self.atm_variance*self.t)
        f = pf/p
        if ascending: 
            cond1, cond2 = (pf>(p_t*f_t)), (f>f_t)
            cond3 = (self.atm_variance*self.t>atmtvar_t)
        else: 
            cond1, cond2 = (pf<(p_t*f_t)), (f<f_t)
            cond3 = (self.atm_variance*self.t<atmtvar_t)
        if cond1 and cond2 and cond3: return 0
        else: return 1

    def total_variance(self, k: float or np.array) -> float or np.array: 
        kminusm = (k-self.m)
        return self.a+self.b*(self.p*kminusm + np.sqrt(kminusm**2+self.s**2))

    def risk_neutral_density(self, k: float or np.array) -> float or np.array: 
        w = self.total_variance(k=k)
        dwdk = self.tool.dwdk(k=k)
        d2wd2k = self.tool.d2wd2k(k=k)
        sq_dwdk = dwdk**2
        term1 = (1-k*dwdk/(2*w))**2
        term2 = .25*sq_dwdk*(.25+1/w)
        return term1 - term2 + .5*d2wd2k

    def implied_variance(self, k: float or np.array) -> float or np.array: 
        return self.total_variance(k=k)/self.t

    def implied_volatility(self, k: float or np.array) -> float or np.array: 
        return np.sqrt(self.implied_variance(k=k))

    def local_volatility(self, k: float or np.array) -> float or np.array: 
        return self.tool.dwdt(k=k)/self.risk_neutral_density(k=k)

    def power_law_params(self) -> dict[str, float]: 
        _gamma, rho = .5, np.sqrt(1-self.vmt/self.vt)
        return {'gamma': _gamma, 'rho': rho, 'nu': self.ut*2*np.sqrt(self.t)/rho}
     
@dataclass
class SurfaceSVI: 

    rho : float
    nu : float = np.nan 
    _gamma : float = np.nan 
    _lambda : float = np.nan 
    ssvi_function : SSVIFunctions = SSVIFunctions.power_law

    def svi(self, atmtvar: np.array, t: np.array)\
        -> StochasticVolatilityInspired: 
        f = np.sqrt(atmtvar)*self.ssvi_parametrization(atmtvar=atmtvar)
        return StochasticVolatilityInspired(
            atm_variance=atmtvar/t , 
            atm_skew=.5*self.rho*f, 
            slope_call_wing=.5*(1-self.rho)*f,
            slope_put_wing=.5*(1+self.rho)*f, 
            min_variance = atmtvar*(1-self.rho**2)/t
        )
    
    def ssvi_parametrization(self,atmtvar: np.array) -> np.array:
        match self.ssvi_function: 
            case SSVIFunctions.heston_like: 
                x = atmtvar*self._lambda 
                return (1-(1-np.exp(-x)/x))/x
            case SSVIFunctions.power_law: 
                return self.nu*(atmtvar**(-self._gamma))
            case SSVIFunctions.power_law2: 
                g = self.gamma 
                return self.nu/((atmtvar**g)*((1+atmtvar)**(1-g)))
                
    def ssvi_derivatives(self,atmtvar: np.array) -> np.array:
        match self.ssvi_function: 
            case SSVIFunctions.heston_like: 
                x = atmtvar*self._lambda  
                return np.exp(-x)*(np.exp(x)-1-x)/(x**2)
            case SSVIFunctions.power_law: 
                g = self._gamma
                return (1-g)*self.ssvi_parametrization(atmtvar=atmtvar)
            case SSVIFunctions.power_law2: pass 
    
    def parameters_check(self) -> int: 
        cond1g = self._gamma>0
        cond2g = self._gamma<1
        cond_gamma = (cond1g and cond2g)
        cond_nu = self.nu > 0
        match self.ssvi_function: 
            case SSVIFunctions.heston_like: 
                cond_model = self._lambda>0 
            case SSVIFunctions.power_law: 
                cond_model = (cond_gamma and cond_nu)
            case SSVIFunctions.power_law2:  
                cond_model = (cond_gamma and cond_nu)
        cond_rho = abs(self.p) < 1
        if cond_rho and cond_model: return 0
        else: return 1
    
    def butterfly_check(self, atmtvar: np.array) -> int: 
        f = atmtvar*self.ssvi_parametrization(atmtvar)
        f2 = atmtvar*self.ssvi_parametrization(atmtvar)**2
        cond1 = (f*(1+np.abs(self.rho))<4)
        cond2 = (f2*(1+np.abs(self.rho))<=4)
        if cond1 and cond2: return 0 
        else: return 1

    def calendar_spread_check(self, atm_tvar: np.array) -> int: 
        is_increasing = np.all(np.diff(atm_tvar) > 0)
        deriv = self.ssvi_derivatives(atmtvar=atm_tvar)
        f = self.ssvi_parametrization(atmtvar=atm_tvar)
        value = f*(1+np.sqrt(1-self.rho**2))/(self.rho**2)
        cond1 = (deriv>=0) 
        cond2 = (deriv<=value)
        cond = (cond1 and cond2)
        if is_increasing and cond: return 0
        else: return 1
    
    def static_arbitrage_check(self, atm_tvar: np.array) -> int: 
        match self.ssvi_function: 
            case SSVIFunctions.heston_like: 
                b_cond = self.butterfly_check(atmtvar=atm_tvar)
                cs_cond = self.calendar_spread_check(atm_tvar=atm_tvar)
                return np.min(b_cond+cs_cond, 1) 
            case SSVIFunctions.power_law: 
                b_cond = self.butterfly_check(atmtvar=atm_tvar)
                cs_cond = self.calendar_spread_check(atm_tvar=atm_tvar)
                return np.min(b_cond+cs_cond, 1) 
            case SSVIFunctions.power_law2: 
                cond = (self.nu*(1+np.abs(self.rho)) <= 2)
                if cond: return 0
                else: return 1
    
    def total_variance(self, atm_tvar: np.array, 
                       k: np.array, t: np.array) -> np.array:
        f = self.ssvi_parametrization(atm_tvar)
        term1 = self.p*f*k
        term2 = np.sqrt((f*k+self.p)**2 + (1-self.p**2))
        return .5*atm_tvar*(1 + term1 + term2)
    
    def implied_variance(self,atm_tvar: np.array, 
                         k: np.array, t: np.array) -> np.array:
        return self.total_variance(atm_tvar=atm_tvar,k=k,t=t)/t 
    
    def implied_volatility(self,atm_tvar: np.array,  
                           k: np.array, t: np.array) -> np.array:
        return np.sqrt(self.implied_variance(atm_tvar=atm_tvar,k=k,t=t))

    def risk_neutral_density(self, atm_tvar: np.array, 
                       k: np.array, t: np.array) -> np.array:
        svi = self.svi(atmtvar=atm_tvar,t=t)
        return svi.risk_neutral_density(k=k)
    
    def local_volatility(self, atm_tvar: np.array, 
                       k: np.array, t: np.array) -> np.array:
        svi = self.svi(atmtvar=atm_tvar,t=t)
        return svi.local_volatility(k=k)