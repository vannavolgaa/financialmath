from dataclasses import dataclass
import numpy as np 
from py_lets_be_rational import implied_volatility_from_a_transformed_rational_guess
from scipy import interpolate
from financialmath.model.blackscholes.closedform import (
    ClosedFormBlackScholesInput, 
    )

@dataclass 
class EuropeanVanillaParityImpliedYield: 
    P : np.array 
    C : np.array 
    S : np.array 
    K : np.array 
    t : np.array 
    r : np.array = None 
    
    def get(self) -> np.array: 
        C, P, S, K, t = self.C, self.P, self.S, self.K, self.t
        if self.r is None: 
            return np.log((C-P-S)/-K)/-t
        else: return np.log((C-P+K*np.exp(-self.r*t))/S)/-t
    
@dataclass
class EuropeanVanillaImpliedVolatility: 
    price : np.array 
    S : np.array 
    K : np.array 
    t : np.array 
    r : np.array 
    q : np.array 
    call : bool = True
    future : bool = True

    def __post_init__(self): 
        if self.future : self.F = self.S 
        else: self.F = self.S*np.exp((self.r-self.q)*self.t)
        if self.call: self.optype = 1 
        else: self.optype = -1
        self.udprice = self.price*np.exp(self.r*self.t)
    
    def get_implied_vol(self, price, F, K, t) -> float: 
        try: 
            return implied_volatility_from_a_transformed_rational_guess(
                price = price, 
                F = F, 
                K = K, 
                T = t, 
                q = self.optype
            )
        except Exception as e: 
            return np.nan
    
    def get(self) -> np.array: 
        if isinstance(self.price, float) or isinstance(self.price, int):
            return self.get_implied_vol(
                price = self.udprice, 
                F = self.F, 
                K = self.K, 
                t = self.t
            ) 
        else: 
            ivs = []
            for i in range(0, len(self.price)): 
                ivs.append(self.get_implied_vol(
                price = self.udprice[i], 
                F = self.F[i], 
                K = self.K[i], 
                t = self.t[i]
            ))
            return np.array(ivs)
            
@dataclass
class EuropeanVanillaImpliedData:
    P : np.array 
    C : np.array 
    S : np.array 
    K : np.array 
    t : np.array 
    r : np.array = None 
    carry_cost : bool = False
    future : bool = False 
    
    def __post_init__(self): 
        if self.future: self.carry_cost = False
        self.n = len(self.P)
        self.pcp = EuropeanVanillaParityImpliedYield(
                    P = self.P, 
                    C = self.C, 
                    S = self.S, 
                    K = self.K, 
                    t = self.t, 
                    r = self.r)
        self.ccts = self.carry_cost_term_structure()
        self.ycts = self.yield_curve_term_structure()
    
    def interpolated_term_structure(self, iyields:np.array)\
        -> interpolate.interp1d: 
        unique_t = list(set(self.t))
        yields = []
        for t in unique_t: 
            tpos = np.where(self.t==t)
            iy = iyields[tpos]
            yields.append(np.mean(iy))
        return interpolate.interp1d(unique_t,yields)
    
    def carry_cost_term_structure(self) -> interpolate.interp1d: 
        if self.r is None or self.carry_cost is False: 
            return interpolate.interp1d([0,100], [0,0])
        else: return self.interpolated_term_structure(self.pcp.get())
                
    def yield_curve_term_structure(self) -> interpolate.interp1d:
        if self.r is None: yields = self.pcp.get()
        else: yields = self.r
        return self.interpolated_term_structure(yields)
       
    def call_implied_volatilies(self) -> np.array: 
        bsiv = EuropeanVanillaImpliedVolatility(
            price = self.C, 
            S = self.S, 
            K = self.K, 
            t = self.t, 
            r = self.ycts(self.t), 
            q = self.ccts(self.t), 
            call = True, 
            future = self.future
        )
        return bsiv.get()
    
    def put_implied_volatilies(self) -> np.array: 
        bsiv = EuropeanVanillaImpliedVolatility(
            price = self.P, 
            S = self.S, 
            K = self.K, 
            t = self.t, 
            r = self.ycts(self.t), 
            q = self.ccts(self.t), 
            call = False, 
            future = self.future
        )
        return bsiv.get()
    
    def implied_volatilities(self) -> np.array: 
        put = self.put_implied_volatilies()
        call = self.call_implied_volatilies()
        pos_nan_put = np.where(np.isnan(put) == True)
        ivs = put 
        ivs[pos_nan_put] = call[pos_nan_put]
        return ivs

    def get(self) -> ClosedFormBlackScholesInput: 
        return ClosedFormBlackScholesInput(
            S = self.S, 
            r = self.ycts(self.t), 
            q = self.ccts(self.t), 
            sigma = self.implied_volatilities(), 
            t = self.t, 
            K = self.K)
    
@dataclass
class AmericanVanillaImpliedData: 
    P : np.array 
    C : np.array 
    S : np.array 
    K : np.array 
    t : np.array 
    r : np.array = None 
    carry_cost : bool = False
    future : bool = False 

    def __post_init__(self): 
        if self.future: self.carry_cost = False
        self.n = len(self.P)
        self.euro = EuropeanVanillaImpliedData(
                    P = self.P, 
                    C = self.C, 
                    S = self.S, 
                    K = self.K, 
                    t = self.t, 
                    r = self.r, 
                    carry_cost=self.carry_cost, 
                    future=self.future)
        self.eurodata = self.euro.get()
    
    def parametric_smile(self, lm: np.array, b0:float, b1:float, b2:float)\
        -> np.array: 
        return b0 + b1*np.log(lm) + b2*np.log(lm)**2
    
    def loss_function(self, x): 
        pass




    
    
    