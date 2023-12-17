import numpy as np
from dataclasses import dataclass
from typing import List
from enum import Enum

@dataclass
class SVITool:
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
        self.sqwt = np.sqrt(self.wt)
        self.ctpt = self.ct+self.pt
        self.b = .5*self.sqwt*self.ctpt   
        self.p = 1 - 2 * self.pt/self.ctpt
        self.beta = self.p - 4*self.ut/self.ctpt
        self.alpha = np.sign(self.beta)*np.sqrt(-1+1/(self.beta**2))
        self.m = self.get_m()
        self.a_and_s = self.get_s_a()
        self.a = self.a_and_s['a']
        self.s = self.a_and_s['s']
        
    def get_m(self) -> float or np.array: 
        a = self.alpha
        p = self.p
        factor_num = self.vt - self.vmt 
        factor_den = -p + np.sign(a) * np.sqrt(1+a**2) - a*np.sqrt(1-p**2)
        factor = factor_num/factor_den
        return factor*self.t/self.b
    
    def is_m_equal_zero(self) -> bool or List[bool]: 
        if not isinstance(self.m, float): 
            return [m == 0 for m in list(self.m)] 
        else: return (self.m == 0)
    
    def get_s_a(self) -> dict[str, float or np.array]: 
        sq1p =  np.sqrt(1-self.p**2)
        s = self.alpha*self.m
        a = self.vmt*self.t - self.b*s*sq1p
        cond = self.is_m_equal_zero()
        a2 = self.t*(self.vmt + self.vt*sq1p)/(1-sq1p)
        s2 = (self.vt*self.t - a2)/self.b 
        if not isinstance(cond, list): 
            if cond: a, s = a2, s2
        else: 
            condpos = np.where(np.array(cond) == True)
            a[condpos] = a2[condpos]
            s[condpos] = s2[condpos]
        return {'a' : a, 's': s}

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
        cond_rho = abs(self.p) < 1
        cond_b = self.b >= 0
        cond_s = self.s > 0
        cond_f = (self.a + self.b*self.s*np.sqrt(1-self.p**2))>0
        if cond_rho & cond_b & cond_s & cond_f: return 0
        else: return 1
    
    def butterfly_check(self) -> int: 
        sqwt = np.sqrt(self.vt*self.t)
        mcptct = np.max([self.ct, self.pt])
        ptct = self.pt+self.ct
        cond1 = (sqwt*mcptct)<2
        cond2 = (ptct*mcptct)<=2
        if cond1 & cond2: return 0 
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
        if cond1 & cond2 & cond3: return 0
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

@dataclass
class SSVIFunctions: 

    @staticmethod
    def heston_like_function(atmtvar: float or np.array, _lambda: float)\
        -> float or np.array: 
        x = atmtvar*_lambda
        return (1-(1-np.exp(-x)/x))/x
    
    @staticmethod
    def power_law_1(atmtvar: float or np.array, _nu: float, _gamma:float)\
        -> float or np.array: 
        return _nu*(atmtvar**(-_gamma))
    
    @staticmethod
    def power_law_2(atmtvar: float or np.array, _nu: float, _gamma:float)\
        -> float or np.array: 
        return _nu/((atmtvar**(_gamma))*((1+atmtvar)**(1-_gamma)))

dataclass
class SurfaceSVI: 

    nu : float 
    rho : float 
    _gamma : float 
    ssvi_function = SSVIFunctions.power_law1

    def svi(self, atmtvar: np.array, t: np.array)\
        -> StochasticVolatilityInspired: 
        f = np.sqrt(atmtvar)*self.power_law(atmtvar=atmtvar)
        return StochasticVolatilityInspired(
            atm_variance=atmtvar/t , 
            atm_skew=.5*self.rho*f, 
            slope_call_wing=.5*(1-self.rho)*f,
            slope_put_wing=.5*(1+self.rho)*f, 
            min_variance = atmtvar*(1-self.rho**2)/t
        )
    
    def parameters_check(self) -> int: 
        cond_gamma = (self._gamma>0) & (self._gamma<1)
        cond_nu = self.nu > 0
        cond_rho = abs(self.p) < 1
        if cond_rho & cond_gamma & cond_nu: return 0
        else: return 1
    
    def butterfly_check(self, atmtvar: np.array) -> int: 
        f = atmtvar*self.power_law(atmtvar)
        f2 = atmtvar*self.power_law(atmtvar)**2
        cond1 = f*(1+np.abs(self.rho))<4
        cond2 = f2*(1+np.abs(self.rho))<=4
        if cond1 & cond2: return 0 
        else: return 1

    def calendar_spread_check(self, f_t: np.array, p_t : np.array, 
                              ascending:bool = True) -> int: 
        pass

    
    def power_law(self, atmtvar: np.array) -> np.array: 
        return self.ssvi_function(atmtvar=atmtvar, _nu=self.nu, 
                                  _gamma = self._gamma)

    def total_variance(self, atm_tvar: np.array, 
                       k: np.array, t: np.array) -> np.array:
        pl = self.power_law(atm_tvar)
        term1 = self.p*pl*k
        term2 = np.sqrt((pl*k+self.p)**2 + (1-self.p**2))
        return .5*atm_tvar*(1 + term1 + term2)
    
    def implied_variance(self,atm_tvar: np.array, 
                         k: np.array, t: np.array) -> np.array:
        return self.total_variance(atm_tvar=atm_tvar,k=k,t=t)/t 
    
    def implied_volatility(self,atm_tvar: np.array,  
                           k: np.array, t: np.array) -> np.array:
        return np.sqrt(self.implied_variance(atm_tvar=atm_tvar,k=k,t=t))

