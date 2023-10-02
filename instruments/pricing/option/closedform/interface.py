from instruments.option import Option, OptionPayoffList
from pricing.option.obj import (ImpliedOptionMarketData, OptionValuationFunction, 
OptionGreeks)
from dataclasses import dataclass
from typing import List
from closedform.framework.blackscholes import (BlackScholesEuropeanVanillaCall, 
                            BlackScholesEuropeanVanillaPut, BlackEuropeanVanilla)


@dataclass
class ClosedFormOptionPricer: 

    marketdata : ImpliedOptionMarketData or List[ImpliedOptionMarketData]
    option : Option or List[Option]
    
    
    def __post_init__(self): 
        if isinstance(self.option, List): 
            self.n = len(option)
            self.multiple = True
        else : self.multiple = False
        self.payoffs = [o.payoff for o in self.option]

    @staticmethod
    def get_greeks(valuationclass:OptionValuationFunction) -> greeks: 

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

    def european_vanilla_call(self): 
        pass

    def european_vanilla_put(self): 
        pass

    def fut_european_vanilla_call(self): 
        pass

    def fut_european_vanilla_put(self): 
        pass