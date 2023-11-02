import numpy as np
from scipy.stats import norm
from typing import List
from financialmath.pricing.option.schema import OptionValuationFunction
from financialmath.quanttool import QuantTool
from dataclasses import dataclass

@dataclass 
class BlackScholesInputData: 
    S : float or list[float]
    F : float or list[float]
    r : float or list[float]
    q : float or list[float]
    sigma : float or list[float]
    t : float or list[float]
    K : float or list[float]

class BlackScholesEuropeanVanillaCall(OptionValuationFunction): 

    def __init__(self, inputdata:BlackScholesInputData):
        self.S = QuantTool.convert_to_numpy_array(x=inputdata.S)
        self.K = QuantTool.convert_to_numpy_array(x=inputdata.K)
        self.q = QuantTool.convert_to_numpy_array(x=inputdata.q)
        self.r = QuantTool.convert_to_numpy_array(x=inputdata.r)
        self.t = QuantTool.convert_to_numpy_array(x=inputdata.t)
        self.sigma = QuantTool.convert_to_numpy_array(x=inputdata.sigma)
        self.d1 = QuantTool.compute_blackscholes_d1(S = self.S,K = self.K,
                                                    r = self.r,q = self.q,
                                                    t = self.t, sigma = self.sigma)
        self.d2 = QuantTool.compute_blackscholes_d2(S = self.S,K = self.K,
                                                    r = self.r,q = self.q,
                                                    t = self.t, sigma = self.sigma)
        self.Nd1 = norm.cdf(self.d1)
        self.Nd2 = norm.cdf(self.d2)
        self.nd1 = norm.pdf(self.d1)
        self.nd2 = norm.pdf(self.d2)

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
    
class BlackScholesEuropeanVanillaPut(OptionValuationFunction): 

    def __init__(self, inputdata:BlackScholesInputData):
        self.S = QuantTool.convert_to_numpy_array(x=inputdata.S)
        self.K = QuantTool.convert_to_numpy_array(x=inputdata.K)
        self.q = QuantTool.convert_to_numpy_array(x=inputdata.q)
        self.r = QuantTool.convert_to_numpy_array(x=inputdata.r)
        self.t = QuantTool.convert_to_numpy_array(x=inputdata.t)
        self.sigma = QuantTool.convert_to_numpy_array(x=inputdata.sigma)
        self.d1 = QuantTool.compute_blackscholes_d1(S = self.S,K = self.K,
                                                    r = self.r,q = self.q,
                                                    t = self.t, sigma = self.sigma)
        self.d2 = QuantTool.compute_blackscholes_d2(S = self.S,K = self.K,
                                                    r = self.r,q = self.q,
                                                    t = self.t, sigma = self.sigma)
        self.Nd1 = norm.cdf(-self.d1)
        self.Nd2 = norm.cdf(-self.d2)
        self.nd1 = norm.pdf(-self.d1)
        self.nd2 = norm.pdf(-self.d2)
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
 
class BlackEuropeanVanillaCall(OptionValuationFunction): 

    def __init__(self, inputdata:BlackScholesInputData):
        self.F = QuantTool.convert_to_numpy_array(x=inputdata.F)
        self.K = QuantTool.convert_to_numpy_array(x=inputdata.K)
        self.r = QuantTool.convert_to_numpy_array(x=inputdata.r)
        self.t = QuantTool.convert_to_numpy_array(x=inputdata.t)
        self.sigma = QuantTool.convert_to_numpy_array(x=inputdata.sigma)
        self.df = np.exp(-self.r*self.t) 
        inputdata_bs = BlackScholesInputData(
            S=self.F,r=0,q=0,t=self.t, sigma=self.sigma, K=self.K)
        self.bs = BlackScholesEuropeanVanillaCall(inputdata_bs) 
    
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
 
class BlackEuropeanVanillaPut(OptionValuationFunction): 

    def __init__(self, inputdata:BlackScholesInputData):
        self.F = QuantTool.convert_to_numpy_array(x=inputdata.F)
        self.K = QuantTool.convert_to_numpy_array(x=inputdata.K)
        self.r = QuantTool.convert_to_numpy_array(x=inputdata.r)
        self.t = QuantTool.convert_to_numpy_array(x=inputdata.t)
        self.sigma = QuantTool.convert_to_numpy_array(x=inputdata.sigma)
        lparam = [len(x) for x in [self.F, self.K, self.r, self.t, self.sigma]]
        self.df = np.exp(-self.r*self.t) 
        inputdata_bs = BlackScholesInputData(
            S=self.F,r=0,q=0,t=self.t, sigma=self.sigma, K=self.K)
        self.bs = BlackScholesEuropeanVanillaPut(inputdata_bs) 
    
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


    






    





    




    
    

    

    