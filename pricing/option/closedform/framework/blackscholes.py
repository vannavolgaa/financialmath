import numpy as np
from scipy.stats import norm
from typing import List

class BlackScholesEuropeanVanilla: 

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
    
    def call_price(self):
        p1 = norm.cdf(self.d1)
        p2 = norm.cdf(self.d2)
        S = self.S
        K = self.K
        t = self.t
        r = self.r
        q = self.q
        F = S*np.exp((r-q)*t)
        return np.exp(-r*t)*(F*p1-K*p2)
  
    def put_price(self):
        p1 = norm.cdf(-self.d1)
        p2 = norm.cdf(-self.d2)
        S = self.S
        K = self.K
        t= self.t
        r = self.r
        q = self.q
        F = S*np.exp((r-q)*t)
        return np.exp(-r*t)*(K*p2-F*p1)

    def call_delta(self):
        q = self.q
        t = self.t
        return np.exp(-q*t)*norm.cdf(self.d1)
    
    def put_delta(self):
        q = self.q
        t = self.t
        return -np.exp(-q*t)* norm.cdf(-self.d1)
    
    def gamma(self):
        S = self.S
        t = self.t
        sigma = self.sigma
        q = self.q
        return np.exp(-q*t)*norm.pdf(self.d1)/(S*sigma*np.sqrt(t))
   
    def vega(self):
        S = self.S
        t = self.t
        q = self.q
        return S*np.exp(-q*t)*norm.pdf(self.d1)*np.sqrt(t)
    
    def call_theta(self):
        S = self.S
        K = self.K
        t= self.t
        r = self.r
        q = self.q
        sigma = self.sigma
        term1 = -np.exp(q*t)*S*norm.pdf(self.d1)*sigma/(2*np.sqrt(t))
        term2 = -r*K*np.exp(-r*t)*norm.cdf(self.d2)
        term3 = q*K*np.exp(-q*t)*norm.cdf(self.d1)
        return  term1 + term2 + term3
    
    def put_theta(self):
        S = self.S
        K = self.K
        t = self.t
        r = self.r
        q = self.q
        sigma = self.sigma
        term1 = -np.exp(q*t)*S*norm.pdf(self.d1)*sigma/(2*np.sqrt(t))
        term2 = r*K*np.exp(-r*t)*norm.cdf(-self.d2)
        term3 = +q*K*np.exp(-q*t)*norm.cdf(-self.d1)
        return  term1 + term2 + term3
    
    def call_rho(self):
        K = self.K
        r = self.r
        t= self.t
        return K*t*np.exp(-r*t)*norm.cdf(self.d2)
        
    def put_rho(self):
        K = self.K
        r = self.r
        t= self.t
        return -K*t*np.exp(-r*t)*norm.cdf(-self.d2)
           
    def call_epsilon(self): 
        S = self.S
        t = self.t
        q = self.q
        return -S*t*np.exp(-q*t)*norm.cdf(self.d1)

    def put_epsilon(self):
        S = self.S
        t = self.t
        q = self.q
        return S*t*np.exp(-q*t)*norm.cdf(-self.d1)

    def call_lambda(self): 
        S = self.S
        return self.call_delta *S/self.call_price

    def put_lambda(self): 
        S = self.S
        return self.put_delta *S/self.put_price
    
    def vanna(self): 
        S = self.S
        t = self.t
        sigma = self.sigma
        return (1-self.d1/(sigma*np.sqrt(t)))*self.vega()/S
    
    def volga(self): 
        sigma = self.sigma
        return self.vega()*self.d1*self.d2/sigma

    def call_dualdelta(self): 
        r = self.r
        t = self.t
        return np.exp(-r*t)*norm.cdf(self.d2)
    
    def put_dualdelta(self): 
        r = self.r
        t = self.t
        return np.exp(-r*t)*norm.cdf(-self.d2)
    
    def dualgamma(self): 
        K = self.K
        t = self.t
        r = self.r
        sigma = self.sigma
        return np.exp(-r*t)*norm.pdf(self.d2)/(K*sigma*np.sqrt(t))

    def speed(self): 
        S = self.S
        t = self.t
        sigma = self.sigma
        return -(self.gamma()/S)*(1+self.d1/(sigma*np.sqrt(t)))

    def call_charm(self): 
        q = self.q 
        t = self.t 
        r = self.r
        sigma = self.sigma
        d1 = self.d1
        d2 = self.d2
        term1 = q*np.exp(-q*t)*norm.cdf(d1)
        term2 = np.exp(-q*t)*norm.pdf(d1)
        term3 = 2*(r-q)*t-d2*sigma*np.sqrt(t)/(2*t*sigma*np.sqrt(t))
        return term1-term2*term3

    def put_charm(self): 
        q = self.q 
        t = self.t 
        r = self.r
        sigma = self.sigma
        d1 = self.d1
        d2 = self.d2
        term1 = q*np.exp(-q*t)*norm.cdf(-d1)
        term2 = np.exp(-q*t)*norm.pdf(d1)
        term3 = 2*(r-q)*t-d2*sigma*np.sqrt(t)/(2*t*sigma*np.sqrt(t))
        return -term1-term2*term3

    def veta(self): 
        q = self.q 
        t = self.t 
        r = self.r
        sigma = self.sigma
        d1 = self.d1
        d2 = self.d2
        S = self.S
        term1 = S*np.exp(-q*t)*norm.pdf(d1)*np.sqrt(t)
        term2 = q+d1*(r-q)/(sigma*np.sqrt(t))-(1+d1*d2)/(2*t)
        return term1*term2

    def color(self): 
        q = self.q 
        t = self.t 
        r = self.r
        sigma = self.sigma
        d1 = self.d1
        d2 = self.d2
        S = self.S
        term1 = 2*S*t*sigma*np.sqrt(t)
        term2 = d1*(2*(r-q)*t-d2*sigma*np.sqrt(t))/(sigma*np.sqrt(t))
        return np.exp(-q*t)*norm.pdf(d1)*(2*q*t+1+term2)/term1

    def zomma(self): 
        gamma = self.gamma()
        d1 = self.d1
        d2 = self.d2
        return gamma * (d1*d2-1)/self.sigma 

    def ultima(self): 
        vega = self.vega()
        d1 = self.d1
        d2 = self.d2
        term1 = -vega / (self.sigma**2)
        term2 = d1*d2*(1-d1*d2) + d1**2 + d2**2
        return term1*term2

class BlackScholesEuropeanBinary: 

    def __init__(self, S: np.array, K: np.array, r: np.array, q: np.array, 
                t: np.array, sigma: np.array):
        
        self.S = S
        self.K = K
        self.q = q
        self.r = r
        self.t = t
        self.sigma = sigma
        self.bs = BlackScholesEuropeanVanilla(S=S,K=K,r=r,q=q,sigma=sigma,t=t)

    def call_price(self): 
        r = self.r 
        t = self.t 
        return np.exp(-r*t)*norm.cdf(self.bs.d2)
    
    def put_price(self): 
        r = self.r 
        t = self.t 
        return np.exp(-r*t)*norm.cdf(-self.bs.d2)
    
    def put_delta(self): 
        r = self.r 
        t = self.t 
        S = self.S 
        sigma = self.sigma 
        d1 = self.bs.d1 
        d2 = self.bs.d2
        return -np.exp(-r*t)*norm.pdf(-d2)/(S*sigma*np.sqrt(t))
    
    def call_delta(self): 
        return - self.put_delta()
    
    def put_gamma(self): 
        t = self.t 
        S = self.S 
        sigma = self.sigma 
        d1 = self.bs.d1 
        factor = d1/(S*sigma*np.sqrt(t))
        return factor*self.put_delta()
    
    def call_gamma(self): 
        return -self.put_gamma()
    
    def put_speed(self): 
        t = self.t 
        S = self.S 
        r=self.r
        sigma = self.sigma 
        d1 = self.bs.d1 
        d2 = self.bs.d2
        term1 = np.exp(-r*t)*norm.pdf(d2)/((sigma**2)*(S**3)*t)
        term2 = -2*d1 + (1-d1*d2)/(sigma*np.sqrt(t))
        return term1*term2
    
    def call_speed(self): 
        return -self.put_speed()
    
    def call_rho(self): 
        t = self.t 
        sigma = self.sigma 
        r=self.r 
        d2 = self.bs.d2
        return np.exp(-r*t)*(-t*norm.cdf(d2) + norm.pdf(d2)*np.sqrt(t)/sigma)

    def put_rho(self): 
        t = self.t 
        sigma = self.sigma 
        r=self.r 
        d2 = self.bs.d2
        return np.exp(-r*t)*(-t*norm.cdf(-d2)- norm.pdf(-d2)*np.sqrt(t)/sigma)

    def call_epsilon(self): 
        t = self.t 
        S = self.S 
        r=self.r 
        d2 = self.bs.d2
        return -t*np.exp(-r*t)*norm.cdf(d2)

    def put_epsilon(self): 
        t = self.t 
        S = self.S 
        r=self.r 
        d2 = self.bs.d2
        return t*np.exp(-r*t)*norm.cdf(-d2)

    def vega(self): 
        t = self.t 
        sigma = self.sigma
        r=self.r 
        d2 = self.bs.d2
        d1 = self.bs.d1
        return np.exp(-r*t)*norm.pdf(-d2)*d1/sigma

    def put_vanna(self): 
        t = self.t 
        sigma = self.sigma
        r=self.r 
        S = self.S
        d2 = self.bs.d2
        d1 = self.bs.d1
        return np.exp(-r*t)*norm.pdf(d2)*(1-d1*d2)/(S*(sigma**2)*np.sqrt(t))
    
    def call_vanna(self): 
        t = self.t 
        sigma = self.sigma
        r=self.r 
        S = self.S
        d2 = self.bs.d2
        d1 = self.bs.d1
        return -np.exp(-r*t)*norm.pdf(d2)*(1-d1*d2)/(S*(sigma**2)*np.sqrt(t))
    
    def call_theta(self): 
        t = self.t 
        sigma = self.sigma
        r=self.r 
        S = self.S
        d2 = self.bs.d2
        d1 = self.bs.d1 
        return np.exp(-r*t)*(r*norm.pdf(d2)+norm.cdf(d2)*(d1/(2*t)-(r-q)/(sigma*np.sqrt(t))))
    
    def put_theta(self): 
        t = self.t 
        sigma = self.sigma
        r=self.r 
        S = self.S
        d2 = self.bs.d2
        d1 = self.bs.d1 
        return np.exp(-r*t)*(r*norm.pdf(-d2)-norm.cdf(-d2)*(d1/(2*t)-(r-q)/(sigma*np.sqrt(t))))
    
    def call_volga(self): 
        t = self.t 
        sigma = self.sigma
        r=self.r 
        S = self.S
        d2 = self.bs.d2
        d1 = self.bs.d1 
        factor = np.exp(-r*t)*((d1**2)*d2-d1-d2)/(sigma**2)
        return factor * norm.cdf(d2)
    
    def put_volga(self): 
        t = self.t 
        sigma = self.sigma
        r=self.r 
        S = self.S
        d2 = self.bs.d2
        d1 = self.bs.d1 
        factor = np.exp(-r*t)*((d1**2)*d2-d1-d2)/(sigma**2)
        return factor * norm.cdf(-d2)
    
class BlackEuropeanVanilla: 

    def __init__(self, F: np.array, K: np.array, r: np.array, 
                t: np.array, sigma: np.array):
                
        self.F = F
        self.K = K
        self.r = r
        self.t = t
        self.sigma = sigma
        self.bs = BlackScholesEuropeanVanilla(S=F,K=K,r=0,q=0,sigma=sigma,t=t)

    def __post_init__(self): 
        self.df = np.exp(-self.r*self.t) 
    def call_price(self): 
        return self.df*self.bs.call_price()
    def put_price(self):
        return self.df*self.bs.put_price()
    def call_delta(self): 
        return self.df*self.bs.call_delta()
    def put_delta(self): 
        return self.df*self.bs.put_delta()
    def vega(self): 
        return self.df*self.bs.vega()
    def put_rho(self): 
        return self.t*self.df*self.put_price()
    def call_rho(self): 
        return self.t*self.df*self.call_price()
    def put_epsilon(self): 
        return 0
    def call_epsilon(self): 
        return 0
    def call_theta(self): 
        return self.df*(self.bs.call_theta() - self.r*self.bs.call_price())
    def put_theta(self): 
        return self.df*(self.bs.put_theta() - self.r*self.bs.put_price()) 
    def gamma(self): 
        return self.df*self.bs.gamma() 
    def vanna(self): 
        return self.df*self.bs.vanna()  
    def volga(self): 
        return self.df*self.bs.volga()  
    def speed(self): 
        return self.df*self.bs.speed()
    def call_charm(self): 
        return self.df*(self.bs.call_charm() - self.r*self.bs.call_delta()) 
    def put_charm(self): 
        return self.df*(self.bs.put_charm() - self.r*self.bs.put_delta()) 
    def veta(self): 
        return self.df*(self.bs.veta() - self.r*self.bs.vega())   
    def zomma(self): 
        return self.df*self.bs.zomma() 
    def ultima(self): 
        return self.df*self.bs.ultima() 
    def color(self): 
        return self.df*(self.bs.color() - self.r*self.bs.gamma())   


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
    
class BlackShcolesEuropeanVanillaCall: 

    def __init__(self, S, K, r, q, sigma, t):

        self.S = S
        self.K = K
        self.q = q
        self.r = r
        self.t = t
        self.sigma = sigma
        self.tool = BlackScholesTool(S=S, K=K, r=r, q=q, sigma=sigma, t=t)
    
    def __post_init__(self): 
        self.d1 = self.tool.compute_d1()
        self.d2 = self.tool.compute_d2()
        self.Nd1 = norm.pdf(self.d1)
        self.Nd2 = norm.pdf(self.d2)
        self.nd1 = norm.cdf(self.d1)
        self.nd2 = norm.cdf(self.d2)

    def price(self): 
        pass 
    def delta(self): 
        pass 
    def vega(self): 
        pass 
    def gamma(self): 
        pass 
    def rho(self): 
        pass 
    def epsilon(self): 
        pass 
    def vanna(self): 
        pass 
    def volga(self): 
        pass 
    def speed(self): 
        pass 
    def charm(self): 
        pass 
    def veta(self): 
        pass 
    def vera(self): 
        pass 
    def zomma(self): 
        pass 
    def ultima(self): 
        pass 
    def main(self): 
        pass 
    


    





    





    




    
    

    

    