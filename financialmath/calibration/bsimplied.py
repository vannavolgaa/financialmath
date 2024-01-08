from dataclasses import dataclass
import numpy as np 
from py_lets_be_rational import implied_volatility_from_a_transformed_rational_guess
from scipy import interpolate, optimize
from financialmath.model.blackscholes.closedform import (
    ClosedFormBlackScholesInput, 
    QuadraticApproximationAmericanVanilla
    )
from financialmath.model.parametricvolatility import ParametricVolatility
from financialmath.calibration.parametricvolatility import LeastSquareRegressionParametricVolatility

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
        if self.future: 
            self.carry_cost = False
        self.t_unique = sorted(np.array(list(set(self.t))))
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
        self.ycts = self.euro.ycts
    
    def loss_function(self, x:np.array, t:float) -> float: 
        tpos = np.where(self.t == t)
        t, K, S, put_price, call_price = t[tpos], K[tpos], S[tpos], \
            self.P[tpos], self.C[tpos]
        if self.r is None: 
            b0, b1, b2, q, r = x[0], x[1], x[2], 0, x[3]
        else: 
            r = self.ycts(t)
            if self.carry_cost: 
                b0, b1, b2, q = x[0], x[1], x[2], x[3]
            else: 
                b0, b1, b2, q = x[0], x[1], x[2], 0
        if self.future: k = K/S
        else: k = K/(S*np.exp((r-q)*t))            
        volmodel = ParametricVolatility(b0 = b0, b1 = b1, b2 = b2)
        inputdata = ClosedFormBlackScholesInput(
            S = S, r = r, q = q, K = K, t = t,
            sigma = volmodel.implied_volatility(k=k, t=t))
        ame_call = QuadraticApproximationAmericanVanilla(
            inputdata = inputdata, future = self.future,  put = False)
        ame_put = QuadraticApproximationAmericanVanilla(
            inputdata = inputdata, future = self.future, put = True)
        model_put_price = ame_put.compute_prices()
        model_call_price = ame_call.compute_prices()
        market = put_price+call_price
        model = model_put_price+model_call_price
        return np.sum((market-model)**2/model)
    
    def calibrate_slice(self, t: float) -> np.array: 
        ed = self.eurodata
        tpos = np.where(self.t == t)
        S, K, sigma, r, q = ed.S[tpos], ed.K[tpos], ed.sigma[tpos],\
            ed.r[tpos], ed.q[tpos]
        if self.future: k = K/S
        else: k = K/(S*np.exp((r-q)*t))
        lrreg = LeastSquareRegressionParametricVolatility(
            ivs = sigma, k=k, t=t, smile=True)
        x_0 = lrreg.coefficients()
        if self.r is None: x_0 = np.insert(x_0, 3, self.ycts(t))  
        else: 
            if self.carry_cost: x_0 = np.insert(x_0, 3, self.ccts(t))  
        fit = optimize.minimize(
            fun = self.loss_function, 
            x0 = x_0, 
            args=(t,), 
            method = 'Nelder-Mead')
        return fit.x
    
    def get(self) -> ClosedFormBlackScholesInput:
        implied_volatilities = np.zeros(self.P.shape)
        yields = np.zeros(self.P.shape)
        carry_cost = np.zeros(self.P.shape)
        for t in self.t_unique: 
            tpos = np.where(self.t == t)
            if len(tpos)<4: 
                nanvec = np.rep(np.nan, len(tpos))
                implied_volatilities[tpos] = nanvec 
                yields[tpos] = nanvec
                carry_cost[tpos] = nanvec
                continue
            cslice = self.calibrate_slice(t=t)
            b0, b1, b2 = cslice[0], cslice[1], cslice[2]
            volmodel = ParametricVolatility(b0 = b0, b1 = b1, b2 = b2)
            if self.r is None: r,q = cslice[3], 0
            else: 
                r = self.ycts(t)
                if self.carry_cost: q = cslice[3]
                else: q = 0  
            K, S = self.K[tpos], self.S[tpos] 
            k = K/(S*np.exp((r-q)*t))
            sigma = volmodel.implied_volatility(k=k, t=t)
            implied_volatilities[tpos] = sigma 
            yields[tpos] = np.rep(r, len(tpos))
            carry_cost[tpos] = np.rep(q, len(tpos))
        return ClosedFormBlackScholesInput(
            S=self.S, 
            r=self.r, 
            q=self.q, 
            sigma=self.implied_volatilities, 
            t=self.t, 
            K=self.K)
            
                     
        
        
            
            
                
    
        
        
        




    
    
    