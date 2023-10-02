from dataclasses import dataclass
from typing import List
import numpy as np
from financialmath.pricing.option.closedform.framework.blackscholes import (
BlackScholesEuropeanVanillaCall,BlackScholesEuropeanVanillaPut, BlackEuropeanVanilla)
from financialmath.instruments.option import Option, OptionPayoffList
from financialmath.pricing.option.obj import (ImpliedOptionMarketData, OptionValuationFunction, 
                                              OptionGreeks, OptionValuationResult)



@dataclass
class ClosedFormOptionPricerObject: 

    payoff: OptionPayoffList
    future : bool 
    options : List[Option]
    marketdata: List[ImpliedOptionMarketData]
    id_number : List[int]

    def __init__(self): 
        self.S = np.array([m.S for m in self.marketdata])
        self.r = np.array([m.r for m in self.marketdata])
        self.q = np.array([m.q for m in self.marketdata])
        self.sigma = np.array([m.sigma for m in self.marketdata])
        self.F = np.array([m.F for m in self.marketdata])
        self.K = np.array([o.specification.strike for o in self.options])
        self.t = np.array([o.specification.tenor.expiry for o in self.options])

    def valuation_class(self): 
        match self.payoff: 
            case OptionPayoffList.european_vanilla_call:
                if self.future:
                    return BlackEuropeanVanilla(
                        F= self.F, K= self.K, r= self.r, 
                        t= self.t, sigma= self.sigma, 
                        Call=True
                    )
                else: 
                    return BlackScholesEuropeanVanillaCall(
                        S= self.S, K= self.K, r= self.r, 
                        q= self.q, t= self.t, sigma= self.sigma, 
                    )
            case OptionPayoffList.european_vanilla_put:
                if self.future:
                    return BlackEuropeanVanilla(
                        F= self.F, K= self.K, r= self.r, 
                        t= self.t, sigma= self.sigma, 
                        Call=False
                    )
                else: 
                    return BlackScholesEuropeanVanillaPut(
                        S= self.S, K= self.K, r= self.r, 
                        q= self.q, t= self.t, sigma= self.sigma, 
                    )
            case OptionPayoffList.european_binary_put:
                return None
            case OptionPayoffList.european_vanilla_call:
                return None
        


@dataclass
class ClosedFormOptionPricer: 

    marketdata : ImpliedOptionMarketData or List[ImpliedOptionMarketData]
    option : Option or List[Option]
    
    def __post_init__(self): 
        self.marketdata = list(self.marketdata)
        self.option = list(self.option)
        self.n = len(self.option)
        self.id_number = list(range(0, self.n-1))
        self.payoffs = [o.payoff for o in self.option]
        self.futopt = [o.specification.forward for o in self.option]
        

    def get_greeks(self, valuationclass:OptionValuationFunction) -> OptionGreeks: 

        delta = valuationclass.delta()
        vega = valuationclass.vega()
        theta = valuationclass.theta()
        rho = valuationclass.rho()
        epsilon = valuationclass.epsilon()
        gamma = valuationclass.gamma()
        vanna = valuationclass.vanna()
        volga = valuationclass.volga()
        ultima = valuationclass.ultima()
        speed = valuationclass.speed()
        zomma = valuationclass.zomma()
        color = valuationclass.color()
        veta = valuationclass.veta()
        vera = valuationclass.vera()
        charm = valuationclass.charm()

        if self.multiple:
            return OptionGreeks(delta=delta, vega=vega, theta=theta, rho=rho, 
                                epsilon=epsilon, gamma=gamma, vanna=vanna, 
                                volga=volga, charm=charm, veta=veta, vera=vera, 
                                speed=speed, zomma=zomma, color=color, 
                                ultima=ultima)
        else: 
            return [OptionGreeks(delta=delta[i], vega=vega[i], 
                                theta=theta[i], rho=rho[i], 
                                epsilon=epsilon[i], gamma=gamma[i], 
                                vanna=vanna[i], volga=volga[i], 
                                charm=charm[i], veta=veta[i], 
                                vera=vera[i], speed=speed[i], 
                                zomma=zomma[i], color=color[i], 
                                ultima=ultima[i]) for i in range(0,self.n-1)]

    def filter_payoff(self, payoff: OptionPayoffList, fut: bool) -> List[bool] or bool: 
        return [p == payoff & f == fut for p,f in zip(self.payoffs, self.futopt)]
        
    def european_vanilla_call(self): 
        payoff_filter = self.filter_payoff(
            payoff=OptionPayoffList.european_vanilla_call, 
            fut = False)
        if self.multiple:
            opts = [o for o,f in zip(self.option,payoff_filter) 
                    if f]
            mda = [m for m,f in zip(self.marketdata,payoff_filter) 
                    if f]
            idn = [i for i,f in zip(self.id_number,payoff_filter) 
                    if f]
        else: 
            if payoff_filter: 
                S, r, q, sigma = self.marketdata.S, self.marketdata.r,\
                                 self.marketdata.q, self.marketdata.sigma
                K = self.option.specification.strike
                t = self.option.specification.tenor.expiry
                valuobj = BlackScholesEuropeanVanillaCall(
                    S=S, K=K, t=t,r=r,q=q, sigma=sigma)
                
            else: 
                pass
       

    def european_vanilla_put(self): 
        pass

    def fut_european_vanilla_call(self): 
        pass

    def fut_european_vanilla_put(self): 
        pass