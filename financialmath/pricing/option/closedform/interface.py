from dataclasses import dataclass
from typing import List
import numpy as np
from financialmath.pricing.option.closedform.framework.blackscholes import (
BlackScholesEuropeanVanillaCall,BlackScholesEuropeanVanillaPut, BlackEuropeanVanilla)
from financialmath.instruments.option import Option, OptionPayoffList, OptionPayoff
from financialmath.pricing.option.obj import (ImpliedOptionMarketData, OptionValuationFunction, 
                                              OptionGreeks, OptionValuationResult)



@dataclass
class ClosedFormOptionPricerObject: 

    payoff : OptionPayoff
    future : bool 
    options : List[Option]
    marketdata: List[ImpliedOptionMarketData]
    id_number : List[int]
    sensitivities : bool

    def __post_init__(self): 
        self.n = len(self.options)
        self.S = np.array([m.S for m in self.marketdata])
        self.r = np.array([m.r for m in self.marketdata])
        self.q = np.array([m.q for m in self.marketdata])
        self.sigma = np.array([m.sigma for m in self.marketdata])
        self.F = np.array([m.F for m in self.marketdata])
        self.K = np.array([o.specification.strike for o in self.options])
        self.t = np.array([o.specification.tenor.expiry for o in self.options])

    def get_valuation_class(self) -> OptionValuationFunction: 
        match self.payoff: 
            case OptionPayoffList.european_vanilla_call.value:
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
            case OptionPayoffList.european_vanilla_put.value:
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
            case OptionPayoffList.european_binary_put.value:
                return None
            case OptionPayoffList.european_vanilla_call.value:
                return None
    
    def get_greeks(self, valuationclass : OptionValuationFunction) -> List[OptionGreeks]: 
        delta = list(valuationclass.delta())
        vega = list(valuationclass.vega())
        theta = list(valuationclass.theta())
        rho = list(valuationclass.rho())
        epsilon = list(valuationclass.epsilon())
        gamma = list(valuationclass.gamma())
        vanna = list(valuationclass.vanna())
        volga = list(valuationclass.volga())
        ultima = list(valuationclass.ultima())
        speed = list(valuationclass.speed())
        zomma = list(valuationclass.zomma())
        color = list(valuationclass.color())
        veta = list(valuationclass.veta())
        #vera = list(valuationclass.vera())
        charm = list(valuationclass.charm())
        return [OptionGreeks(delta=delta[i], vega=vega[i], 
                            theta=theta[i], rho=rho[i], 
                            epsilon=epsilon[i], gamma=gamma[i], 
                            vanna=vanna[i], volga=volga[i], 
                            charm=charm[i], veta=veta[i], 
                            vera=np.nan, speed=speed[i], 
                            zomma=zomma[i], color=color[i], 
                            ultima=ultima[i]) for i in range(0,self.n)]

    def main(self) -> dict: 
        if self.n==0: return dict()
        else:
            valuationclass = self.get_valuation_class()
            price = list(valuationclass.price())
            if self.sensitivities: greeks = self.get_greeks(valuationclass=valuationclass)
            else: greeks = [OptionGreeks() for i in range(0,self.n)]
            method = valuationclass.method()
            result = [OptionValuationResult(instrument=i, price=p, sensitivities=g, 
                                            method = method, marketdata=m) 
                                            for i, p, g, m in 
                                            zip(self.options, price, greeks, 
                                            self.marketdata)]
          
            
            return dict(zip(self.id_number, result))

@dataclass
class ClosedFormOptionPricer: 

    marketdata : ImpliedOptionMarketData or List[ImpliedOptionMarketData]
    option : Option or List[Option]
    sensitivities = True
    
    def __post_init__(self): 
        if not isinstance(self.option, list):
            self.marketdata = [self.marketdata]
            self.option = [self.option]
        self.n = len(self.option)
        self.id_number = list(range(0, self.n))
        self.payoffs = [o.payoff for o in self.option]
        self.futopt = [o.specification.forward for o in self.option]
        
    def get_optionpricer_object(self, payoff: OptionPayoff, fut: bool) -> ClosedFormOptionPricerObject: 
        payoff_filter = [(o.payoff == payoff) & (o.specification.forward == fut)  
                        for o in self.option]
        opt = [o for o,f in zip(self.option,payoff_filter) if f]
        mda = [m for m,f in zip(self.marketdata,payoff_filter) if f]
        idn = [i for i,f in zip(self.id_number,payoff_filter) if f]
        return ClosedFormOptionPricerObject(payoff=payoff, future=fut, 
                                            options=opt, 
                                            marketdata=mda,
                                            id_number=idn, 
                                            sensitivities=self.sensitivities)
                                                  
    def european_vanilla_call(self) -> dict: 
        return self.get_optionpricer_object(
            payoff = OptionPayoffList.european_vanilla_call.value,
            fut = False).main()
       
    def european_vanilla_put(self) -> dict: 
        return self.get_optionpricer_object(
            payoff = OptionPayoffList.european_vanilla_put.value,
            fut = False).main()

    def fut_european_vanilla_call(self) -> dict: 
        return self.get_optionpricer_object(
            payoff = OptionPayoffList.european_vanilla_call.value,
            fut = True
            ).main()
       
    def fut_european_vanilla_put(self) -> dict: 
        return self.get_optionpricer_object(
            payoff = OptionPayoffList.european_vanilla_put.value,
            fut = True
            ).main()

    def manage_unpriced_options(self, dict_result: dict) -> dict:
        priced_option_ids =  list(dict_result.keys())
        unpriced_options_filter = [i in priced_option_ids for i in self.id_number]
        unpriced_options = [o for o, f in zip(self.option, unpriced_options_filter)
                            if f is False]
        unused_market_data = [m for m, f in zip(self.marketdata, unpriced_options_filter)
                            if f is False]
        unpriced_options_ids = [i for i, f in zip(self.id_number, unpriced_options_filter)
                            if f is False]
        result_unpriced = [OptionValuationResult(instrument = o, marketdata = m,
                                                price = np.nan, sensitivities=False, 
                                                method = None) 
                                                for o, m in zip(
                                                    unpriced_options,
                                                    unused_market_data
                                                    )]
        return dict(zip(unpriced_options_ids, result_unpriced))

    def manage_priced_options(self) -> dict: 
        output = dict()
        result_list = [self.european_vanilla_call(), self.european_vanilla_put(), 
                    self.fut_european_vanilla_call(), self.fut_european_vanilla_put()]
        for r in result_list: 
            output.update(r)
        return output
    
    def main(self) -> List[OptionValuationResult] or OptionValuationResult: 
        result = dict()
        priced_options = self.manage_priced_options()
        unpriced_options = self.manage_unpriced_options(dict_result=priced_options)
        result.update(priced_options)
        result.update(unpriced_options)
        result = dict(sorted(result.items()))
        output = list(result.values())
        if len(output)==1: return output[0]
        else: return output