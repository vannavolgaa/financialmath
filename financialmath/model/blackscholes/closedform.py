import numpy as np
from dataclasses import dataclass
from scipy.optimize import newton
from financialmath.tools.probability import NormalDistribution

@dataclass 
class ClosedFormBlackScholesInput: 
    S : float or np.array
    r : float or np.array
    q : float or np.array
    sigma : float or np.array
    t : float or np.array
    K : float or np.array

    def d1(self):
        S, K, r, q, sigma, t = self.S,self.K,self.r,self.q,self.sigma,self.t
        return(np.log(S/K)+(r-q+sigma**2/2)*t)/(sigma*np.sqrt(t))
    
    def d2(self):
        sigma, t = self.sigma,self.t
        return self.d1()-sigma*np.sqrt(t)

class BlackScholesEuropeanVanillaCall: 

    def __init__(self, inputdata:ClosedFormBlackScholesInput):
        self.S, self.K, self.q, self.r, self.t, self.sigma = inputdata.S,\
             inputdata.K,inputdata.q,inputdata.r,inputdata.t,inputdata.sigma
        self.d1 = inputdata.d1()
        self.d2 = inputdata.d2()
        self.Nd1 = NormalDistribution().cdf(self.d1)
        self.Nd2 = NormalDistribution().cdf(self.d2)
        self.nd1 = NormalDistribution().pdf(self.d1)
        self.nd2 = NormalDistribution().pdf(self.d2)

    def get_max_length_param(self) -> int: 
        lparam = [len(x) for x in [self.S, self.K, self.q, 
                                   self.r, self.t, self.sigma]]
        return max(lparam)
    
    def method(self) -> str: 
        return 'Black Scholes closed form vanilla european call option formula'
    
    def price(self) -> float or np.array: 
        S = self.S
        K = self.K
        t = self.t
        r = self.r
        q = self.q
        Nd1 = self.Nd1
        Nd2 = self.Nd2
        F = S*np.exp((r-q)*t)
        return np.exp(-r*t)*(F*Nd1-K*Nd2) 
    
    def delta(self) -> float or np.array: 
        q = self.q
        t = self.t
        return np.exp(-q*t)*self.Nd1
    
    def vega(self) -> float or np.array: 
        S = self.S
        t = self.t
        q = self.q
        return S*np.exp(-q*t)*self.nd1*np.sqrt(t) 
    
    def gamma(self) -> float or np.array: 
        S = self.S
        t = self.t
        sigma = self.sigma
        q = self.q
        return np.exp(-q*t)*self.nd1/(S*sigma*np.sqrt(t)) 
    
    def rho(self) -> float or np.array: 
        K = self.K
        r = self.r
        t= self.t
        return K*t*np.exp(-r*t)*self.Nd2 
    
    def epsilon(self) -> float or np.array: 
        S = self.S
        t = self.t
        q = self.q
        return -S*t*np.exp(-q*t)*self.Nd1
    
    def theta(self) -> float or np.array: 
        S = self.S
        K = self.K
        t= self.t
        r = self.r
        q = self.q
        sigma = self.sigma
        term1 = -np.exp(-q*t)*S*self.nd1*sigma/(2*np.sqrt(t))
        term2 = -r*K*np.exp(-r*t)*self.Nd2
        term3 = q*S*np.exp(-q*t)*self.Nd1
        return  term1 + term2 + term3
    
    def vanna(self) -> float or np.array: 
        S = self.S
        t = self.t
        sigma = self.sigma
        return (1-self.d1/(sigma*np.sqrt(t)))*self.vega()/S 
    
    def volga(self) -> float or np.array: 
        sigma = self.sigma
        return self.vega()*self.d1*self.d2/sigma 
    
    def speed(self) -> float or np.array: 
        S = self.S
        t = self.t
        sigma = self.sigma
        return -(self.gamma()/S)*(1+self.d1/(sigma*np.sqrt(t))) 
    
    def charm(self) -> float or np.array: 
        q = self.q 
        t = self.t 
        r = self.r
        sigma = self.sigma
        d1 = self.d1
        d2 = self.d2
        term1 = q*np.exp(-q*t)*self.Nd1
        term2 = np.exp(-q*t)*self.nd1
        term3 = 2*(r-q)*t-d2*sigma*np.sqrt(t)/(2*t*sigma*np.sqrt(t))
        return term1-term2*term3 
    
    def veta(self) -> float or np.array: 
        q = self.q 
        t = self.t 
        r = self.r
        sigma = self.sigma
        d1 = self.d1
        d2 = self.d2
        S = self.S
        term1 = S*np.exp(-q*t)*self.nd1*np.sqrt(t)
        term2 = q+d1*(r-q)/(sigma*np.sqrt(t))-(1+d1*d2)/(2*t)
        return term1*term2 
    
    def vera(self) -> float or np.array: 
        n = self.get_max_length_param()
        return np.repeat(np.nan,n)
    
    def color(self) -> float or np.array: 
        q = self.q 
        t = self.t 
        r = self.r
        sigma = self.sigma
        d1 = self.d1
        d2 = self.d2
        S = self.S
        term1 = 2*S*t*sigma*np.sqrt(t)
        term2 = d1*(2*(r-q)*t-d2*sigma*np.sqrt(t))/(sigma*np.sqrt(t))
        return np.exp(-q*t)*self.nd1*(2*q*t+1+term2)/term1
    
    def zomma(self) -> float or np.array: 
        gamma = self.gamma()
        d1 = self.d1
        d2 = self.d2
        return gamma * (d1*d2-1)/self.sigma  
    
    def ultima(self) -> float or np.array: 
        vega = self.vega()
        d1 = self.d1
        d2 = self.d2
        term1 = -vega / (self.sigma**2)
        term2 = d1*d2*(1-d1*d2) + d1**2 + d2**2
        return term1*term2 
    
class BlackScholesEuropeanVanillaPut: 

    def __init__(self, inputdata:ClosedFormBlackScholesInput):
        self.S, self.K, self.q, self.r, self.t, self.sigma = inputdata.S,\
             inputdata.K,inputdata.q,inputdata.r,inputdata.t,inputdata.sigma
        self.d1 = inputdata.d1()
        self.d2 = inputdata.d2()
        self.Nd1 = NormalDistribution().cdf(-self.d1)
        self.Nd2 = NormalDistribution().cdf(-self.d2)
        self.nd1 = NormalDistribution().pdf(-self.d1)
        self.nd2 = NormalDistribution().pdf(-self.d2)
        self.eurocall = BlackScholesEuropeanVanillaCall(inputdata=inputdata)

    def get_max_length_param(self) -> int: 
        lparam = [len(x) for x in [self.S, self.K, self.q, 
                                   self.r, self.t, self.sigma]]
        return max(lparam)

    def method(self) -> str: 
        return 'Black Scholes closed form vanilla european put option formula'
    
    def price(self) -> float or np.array: 
        S = self.S
        K = self.K
        t = self.t
        r = self.r
        q = self.q
        Nd1 = self.Nd1
        Nd2 = self.Nd2
        F = S*np.exp((r-q)*t)
        return np.exp(-r*t)*(K*Nd2-F*Nd1) 
    
    def delta(self) -> float or np.array: 
        q = self.q
        t = self.t
        return -np.exp(-q*t)*self.Nd1
    
    def vega(self) -> float or np.array: 
        return self.eurocall.vega()
    
    def gamma(self) -> float or np.array: 
        return self.eurocall.gamma()
    
    def rho(self) -> float or np.array: 
        K = self.K
        r = self.r
        t= self.t
        return -K*t*np.exp(-r*t)*self.Nd2 
    
    def epsilon(self) -> float or np.array: 
        S = self.S
        t = self.t
        q = self.q
        return S*t*np.exp(-q*t)*self.Nd1
    
    def theta(self) -> float or np.array: 
        S = self.S
        K = self.K
        t= self.t
        r = self.r
        q = self.q
        sigma = self.sigma
        term1 = -np.exp(-q*t)*S*self.eurocall.nd1*sigma/(2*np.sqrt(t))
        term2 = r*K*np.exp(-r*t)*self.Nd2
        term3 = -q*S*np.exp(-q*t)*self.Nd1
        return  term1 + term2 + term3
    
    def vanna(self) -> float or np.array: 
        return self.eurocall.vanna()
    
    def volga(self) -> float or np.array: 
        return self.eurocall.volga()
    
    def speed(self) -> float or np.array: 
        return self.eurocall.speed()
    
    def charm(self) -> float or np.array: 
        q = self.q 
        t = self.t 
        r = self.r
        sigma = self.sigma
        d1 = self.d1
        d2 = self.d2
        term1 = -q*np.exp(-q*t)*self.Nd1
        term2 = np.exp(-q*t)*self.eurocall.nd1
        term3 = 2*(r-q)*t-d2*sigma*np.sqrt(t)/(2*t*sigma*np.sqrt(t))
        return term1-term2*term3 
    
    def veta(self) -> float or np.array: 
        return self.eurocall.veta()
    
    def vera(self) -> float or np.array: 
        n = self.get_max_length_param()
        return np.repeat(np.nan,n)
    
    def color(self) -> float or np.array: 
        return self.eurocall.color()
    
    def zomma(self) -> float or np.array: 
        return self.eurocall.zomma()
    
    def ultima(self) -> float or np.array: 
        return self.eurocall.ultima()
 
class BlackEuropeanVanillaCall: 

    def __init__(self, inputdata:ClosedFormBlackScholesInput):
        self.F, self.K, self.r, self.t, self.sigma = inputdata.S,\
             inputdata.K,inputdata.r,inputdata.t,inputdata.sigma
        self.df = np.exp(-self.r*self.t) 
        dbs = ClosedFormBlackScholesInput(
            S = self.F,
            r = 0,
            q = 0,
            t = self.t,
            sigma = self.sigma,
            K = self.K)
        self.bs = BlackScholesEuropeanVanillaCall(dbs) 
    
    def get_max_length_param(self) -> int: 
        lparam = [len(x) for x in [self.F, self.K, self.r, self.t, self.sigma]]
        return max(lparam)

    def method(self) -> str: 
        return 'Black closed form formula'
    
    def price(self) -> float or np.array: 
        return self.df*self.bs.price()
    
    def delta(self) -> float or np.array: 
        return self.df*self.bs.delta()
    
    def vega(self) -> float or np.array: 
        return self.df*self.bs.vega()
    
    def gamma(self) -> float or np.array: 
        return self.df*self.bs.gamma()
    
    def rho(self) -> float or np.array: 
        return self.t*self.df*self.price()
    
    def epsilon(self) -> float or np.array: 
        n = self.get_max_length_param()
        return np.repeat(0, n)
    
    def theta(self) -> float or np.array: 
        return self.df*(self.bs.theta() - self.r*self.bs.price())
    
    def vanna(self) -> float or np.array: 
        return self.df*self.bs.vanna()
    
    def volga(self) -> float or np.array: 
        return self.df*self.bs.volga()
    
    def speed(self) -> float or np.array: 
        return self.df*self.bs.speed()
    
    def charm(self) -> float or np.array: 
        return self.df*(self.bs.charm() - self.r*self.bs.delta()) 
    
    def veta(self) -> float or np.array: 
        return self.df*(self.bs.veta() - self.r*self.bs.vega())  
    
    def vera(self) -> float or np.array: 
        return self.t*self.df*self.vega()
    
    def color(self) -> float or np.array: 
        return self.df*(self.bs.color() - self.r*self.bs.gamma())  
     
    def zomma(self) -> float or np.array: 
        return self.df*self.bs.zomma()
    
    def ultima(self) -> float or np.array: 
        return self.df*self.bs.ultima()
 
class BlackEuropeanVanillaPut: 

    def __init__(self, inputdata:ClosedFormBlackScholesInput):
        self.F, self.K, self.r, self.t, self.sigma = inputdata.S,\
             inputdata.K,inputdata.r,inputdata.t,inputdata.sigma
        self.df = np.exp(-self.r*self.t) 
        dbs = ClosedFormBlackScholesInput(
            S = self.F,
            r = 0,
            q = 0,
            t = self.t,
            sigma = self.sigma,
            K = self.K)
        self.bs = BlackScholesEuropeanVanillaPut(dbs) 
    
    def get_max_length_param(self) -> int: 
        lparam = [len(x) for x in [self.F, self.K, self.r, self.t, self.sigma]]
        return max(lparam)

    def method(self) -> str: 
        return 'Black closed form formula'
    
    def price(self) -> float or np.array: 
        return self.df*self.bs.price()
    
    def delta(self) -> float or np.array: 
        return self.df*self.bs.delta()
    
    def vega(self) -> float or np.array: 
        return self.df*self.bs.vega()
    
    def gamma(self) -> float or np.array: 
        return self.df*self.bs.gamma()
    
    def rho(self) -> float or np.array: 
        return self.t*self.df*self.price()
    
    def epsilon(self) -> float or np.array: 
        n = self.get_max_length_param()
        return np.repeat(0, n)
    
    def theta(self) -> float or np.array: 
        return self.df*(self.bs.theta() - self.r*self.bs.price())
    
    def vanna(self) -> float or np.array: 
        return self.df*self.bs.vanna()
    
    def volga(self) -> float or np.array: 
        return self.df*self.bs.volga()
    
    def speed(self) -> float or np.array: 
        return self.df*self.bs.speed()
    
    def charm(self) -> float or np.array: 
        return self.df*(self.bs.charm() - self.r*self.bs.delta()) 
    
    def veta(self) -> float or np.array: 
        return self.df*(self.bs.veta() - self.r*self.bs.vega())  
    
    def vera(self) -> float or np.array: 
        return self.t*self.df*self.vega()
    
    def color(self) -> float or np.array: 
        return self.df*(self.bs.color() - self.r*self.bs.gamma())  
     
    def zomma(self) -> float or np.array: 
        return self.df*self.bs.zomma()
    
    def ultima(self) -> float or np.array: 
        return self.df*self.bs.ultima()

class QuadraticApproximationAmericanVanilla: 
    
    def __init__(self, inputdata:ClosedFormBlackScholesInput, 
                 future:bool = False, put:bool = False):
        self.future, self.put = future, put
        self.S, self.K, self.q, self.r, self.t, self.sigma = inputdata.S,\
             inputdata.K,inputdata.q,inputdata.r,inputdata.t,inputdata.sigma
        if future: self.b = 0 
        else: self.b = self.r-self.q 
        self.bs = self.return_black_scholes(inputdata)
        self.F = 1-np.exp(-self.r*self.t)
        self.N = 2*self.b/self.sigma**2
        self.M = 2*self.r/self.sigma**2
        self.q2 = (-(self.N-1)+np.sqrt((self.N-1)**2+4*self.M/self.F))/2
        self.q1 = (-(self.N-1)-np.sqrt((self.N-1)**2+4*self.M/self.F))/2
        self.q2_inf = (-(self.N-1)+np.sqrt((self.N-1)**2+4*self.M))/2
        self.q1_inf = (-(self.N-1)-np.sqrt((self.N-1)**2+4*self.M))/2
        self.S_star = self.find_optimal_exercise_price()
        self.euro_prices = self.bs.price()
    
    def return_black_scholes(self, inputdata: ClosedFormBlackScholesInput):
        if self.future: 
            if self.put: return BlackEuropeanVanillaPut(inputdata)
            else: return BlackEuropeanVanillaCall(inputdata) 
        else: 
            if self.put: return BlackScholesEuropeanVanillaPut(inputdata)
            else: return BlackScholesEuropeanVanillaCall(inputdata) 
    
    def initial_optimal_exercise_price(self) -> np.array: 
        K, b, sigma, t = self.K, self.b, self.sigma, self.t
        if self.put: 
            S_star_inf = K/(1-1/self.q1_inf)  
            h = (b*t-2*sigma*np.sqrt(t))*K/(K-S_star_inf)
            return S_star_inf+(K-S_star_inf)*np.exp(h)
        else: 
            S_star_inf = K/(1-1/self.q2_inf)  
            h = -(b*t+2*sigma*np.sqrt(t))*K/(S_star_inf-K)
            return K+(S_star_inf-K)*(1-np.exp(h))
    
    def minimize_function(self, S: np.array) -> np.array:
        inputdata = ClosedFormBlackScholesInput(
            S = S, r = self.r, q = self.q, 
            sigma = self.sigma, t = self.t, K = self.K
        )
        bs = self.return_black_scholes(inputdata=inputdata)
        price, delta = bs.price(), bs.delta()
        if self.put: return self.K - S - price + S*(1+delta)/self.q1
        else: return S - self.K - price - S*(1-delta)/self.q2
    
    def minimize_function_derivative(self, S: np.array) -> np.array:
        inputdata = ClosedFormBlackScholesInput(
            S = S, r = self.r, q = self.q, 
            sigma = self.sigma, t = self.t, K = self.K
        )
        bs = self.return_black_scholes(inputdata=inputdata)
        gamma, delta = bs.gamma(), bs.delta()
        if self.put: 
            ratio = (1/self.q1)
            return -1 - delta*(ratio-1) + ratio*(1+S*gamma)
        else: 
            ratio = (1/self.q2)
            return 1 - delta*(1-ratio) - ratio*(1-S*gamma)
    
    def find_optimal_exercise_price(self) -> np.array:
        return newton(
            func = self.minimize_function,
            fprime = self.minimize_function_derivative, 
            x0 = self.initial_optimal_exercise_price())

    def early_exercise(self) -> np.array: 
        if self.put: return self.K - self.S
        else: return self.S - self.K
    
    def exercise_premium(self) -> np.array: 
        inputdata = ClosedFormBlackScholesInput(
            S = self.S_star, 
            r = self.r, 
            q = self.q, 
            sigma = self.sigma, 
            t = self.t, 
            K = self.K
        )
        bs = self.return_black_scholes(inputdata=inputdata)
        delta = bs.delta()
        if self.put: 
            factor = self.S_star*((self.S/self.S_star)**self.q1)
            return -factor*(1+delta)/self.q1
        else: 
            factor = self.S_star*((self.S/self.S_star)**self.q2)
            return factor*(1-delta)/self.q2
    
    def vectorized_compute_prices(self, early_ex:np.array, 
                                  exprem:np.array, 
                                  early_ex_cond: np.array) -> np.array: 
        exprem = np.maximum(exprem,0)
        indexes = np.where(early_ex_cond)[0]
        inv_cond = np.logical_not(early_ex_cond)
        inv_indexes =  np.where(inv_cond)[0]
        nan_index = np.where(np.isnan(self.S_star))[0]
        result = np.zeros(self.S_star.shape)
        try : result[indexes] = early_ex[indexes]
        except TypeError: 
            early_ex = np.repeat(early_ex, len(self.S_star))
            result[indexes] = early_ex[indexes]
        result[inv_indexes] = self.euro_prices + exprem[inv_indexes]
        result[nan_index] = np.repeat(self.euro_prices, len(nan_index))
        return result
    
    def early_exercise_cond(self) -> np.array: 
        if self.put: return (self.S<=self.S_star)
        else: return (self.S>=self.S_star)

    def compute_prices(self) -> np.array: 
        early_ex = self.early_exercise()
        exercise_premium = self.exercise_premium()
        early_exercise_cond = self.early_exercise_cond()
        if isinstance(self.S_star, float):
            if early_exercise_cond: result = early_ex
            else: 
                if np.isnan(exercise_premium): exercise_premium=0
                result = self.euro_prices + exercise_premium#np.max([exercise_premium,0])
        else: 
            result = self.vectorized_compute_prices(
                early_ex=early_ex, 
                exprem=exercise_premium,
                early_ex_cond=early_exercise_cond
            )
        return result 
        



    
    
    
    





    





    




    
    

    

    