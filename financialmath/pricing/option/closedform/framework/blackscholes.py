import numpy as np
from scipy.stats import norm
from typing import List
from financialmath.pricing.option.obj import OptionValuationFunction, OptionValuation

class BlackScholesTool: 

    def __init__(self, S, K, r, q, sigma, t):

        self.S = S
        self.K = K
        self.q = q
        self.r = r
        self.t = t
        self.sigma = sigma

    def __post_init__(self): 
        self.d1 = self.compute_d1()
        self.d2 = self.compute_d2()

    def compute_d1(self):
        S = self.S
        K = self.K
        r = self.r
        q = self.q
        t = self.t
        sigma = self.sigma
        return(np.log(S/K)+(r-q+sigma**2/2)*t)/(sigma*np.sqrt(t))
   
    def compute_d2(self):
        t = self.t
        sigma = self.sigma
        return self.d1()-sigma*np.sqrt(t)
    
class BlackScholesEuropeanVanillaCall(OptionValuationFunction): 

    def __init__(self, S: float or np.array, 
                K: float or np.array, 
                r: float or np.array, 
                q: float or np.array, 
                sigma: float or np.array, 
                t: float or np.array):
        self.S = S
        self.K = K
        self.q = q
        self.r = r
        self.t = t
        self.sigma = sigma
        self.tool = BlackScholesTool(S=S, K=K, r=r, q=q, sigma=sigma, t=t)
        self.d1 = self.tool.compute_d1()
        self.d2 = self.tool.compute_d2()
        self.Nd1 = norm.cdf(self.d1)
        self.Nd2 = norm.cdf(self.d2)
        self.nd1 = norm.pdf(self.d1)
        self.nd2 = norm.pdf(self.d2)
        self.method = 'Black Scholes closed form formula'
    def method(self) -> str: 
        return 'Black Scholes closed form formula'
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
        return np.nan 
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
    
class BlackScholesEuropeanVanillaPut(OptionValuationFunction): 

    def __init__(self, S: float or np.array, 
                K: float or np.array, 
                r: float or np.array, 
                q: float or np.array, 
                sigma: float or np.array, 
                t: float or np.array):
        self.S = S
        self.K = K
        self.q = q
        self.r = r
        self.t = t
        self.sigma = sigma
        self.tool = BlackScholesTool(S=S, K=K, r=r, q=q, sigma=sigma, t=t)
        self.d1 = self.tool.compute_d1()
        self.d2 = self.tool.compute_d2()
        self.Nd1 = norm.cdf(-self.d1)
        self.Nd2 = norm.cdf(-self.d2)
        self.nd1 = norm.pdf(-self.d1)
        self.nd2 = norm.pdf(-self.d2)
        self.eurocall = BlackScholesEuropeanVanillaCall(S=S, K=K, r=r, q=q, 
                                                    t=t, sigma=sigma)
    def method(self) -> str: 
        return 'Black Scholes closed form formula'
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
        return np.nan 
    def color(self) -> float or np.array: 
        return self.eurocall.color()
    def zomma(self) -> float or np.array: 
        return self.eurocall.zomma()
    def ultima(self) -> float or np.array: 
        return self.eurocall.ultima()
    
class BlackEuropeanVanilla(OptionValuationFunction): 
    def __init__(self, F: float or np.array, 
                K: float or np.array, 
                r: float or np.array, 
                sigma: float or np.array, 
                t: float or np.array, Call = True):

        self.df = np.exp(-self.r*self.t) 

        if Call: 
            self.bs = BlackScholesEuropeanVanillaCall(S=F, K=K, r=0, q=0, 
                                                    t=t, sigma=sigma)  
        else: 
            self.bs = BlackScholesEuropeanVanillaPut(S=F, K=K, r=0, q=0, 
                                                    t=t, sigma=sigma) 
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
        return 0
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


    






    





    




    
    

    

    