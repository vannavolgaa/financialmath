from dataclasses import dataclass
from enum import Enum
from typing import List
from financialmath.pricing.option.closedform.framework.blackscholes import *
from financialmath.instruments.option import (Option, OptionPayoff, OptionalityType, 
    ExerciseType)
from financialmath.pricing.option.schema import (ImpliedOptionMarketData, OptionValuationFunction, 
                                              OptionGreeks, OptionValuationResult)


class PayoffClosedFrom(Enum): 
    european_vanilla_call = OptionPayoff(option_type=OptionalityType.call,
                            exercise=ExerciseType.european) 
    european_vanilla_put = OptionPayoff(option_type=OptionalityType.put, 
                            exercise=ExerciseType.european)   
    fut_european_vanilla_call = OptionPayoff(option_type=OptionalityType.call,
                            exercise=ExerciseType.european, future=True) 
    fut_european_vanilla_put = OptionPayoff(option_type=OptionalityType.put, 
                            exercise=ExerciseType.european, future=True)

    def find_payoff(payoff : OptionPayoff): 
        try : return [pcf for pcf in list(PayoffClosedFrom) 
                        if pcf.value == payoff][0]
        except Exception as e: None

class ValuationMethodClosedFormMapping(Enum): 
    european_vanilla_call = BlackScholesEuropeanVanillaCall
    european_vanilla_put = BlackScholesEuropeanVanillaPut
    fut_european_vanilla_call = BlackEuropeanVanillaCall
    fut_european_vanilla_put = BlackEuropeanVanillaPut

    def find_method(payoff_type: PayoffClosedFrom): 
        try : return [v.value for v in list(ValuationMethodClosedFormMapping) 
                        if v.name == payoff_type.name][0]
        except Exception as e: return None

@dataclass
class ClosedFormOptionPricerObject: 

    payoff_type : PayoffClosedFrom
    options : List[Option]
    marketdata: List[ImpliedOptionMarketData]
    sensitivities : bool

    def __post_init__(self): 
        self.n = len(self.options)
        self.inputdata = BlackScholesInputData(
            S = [m.S for m in self.marketdata], 
            K = [o.specification.strike for o in self.options], 
            r = [m.r for m in self.marketdata], 
            q = [m.q for m in self.marketdata], 
            F = [m.F for m in self.marketdata], 
            t = [o.specification.tenor.expiry for o in self.options], 
            sigma = [m.sigma for m in self.marketdata])
        self.valuation = self.valuation_object()

    def valuation_object(self) -> OptionValuationFunction:
        p = self.payoff_type
        method = ValuationMethodClosedFormMapping.find_method(p)
        return method(self.inputdata)
    
    def main(self) -> List[OptionValuationResult]: 
        instruments = self.instruments
        marketdata = self.inputdata
        price = self.valuation.get_price(n=self.n)
        if self.sensitivities:
            sensi = self.valuation.get_greeks(n=self.n)
        else: sensi = [OptionGreeks()]*self.n
        method = self.valuation.get_method(n=self.n)
        data = zip(instruments, marketdata, price, sensi, method)
        return [OptionValuationResult(i,d,p,s,m) for i,d,p,s,m in data]

@dataclass
class ClosedFormOptionPricer:
    marketdata : ImpliedOptionMarketData or List[ImpliedOptionMarketData]
    option : Option or List[Option]
    sensitivities : bool = True

    def __post_init__(self): 
        if not isinstance(self.option, list):
            self.marketdata = [self.marketdata]
            self.option = [self.option]
        self.n = len(self.option)
        self.id_number = list(range(0, self.n))
        self.payoffs = [o.payoff for o in self.option]
        self.closed_form_payoff = [PayoffClosedFrom.find_payoff(payoff=p)
                                    for p in self.payoffs]
    
    def pricing(self, payoff_type: PayoffClosedFrom) -> dict: 
        myfilter = [(clp == payoff_type) for clp in self.closed_form_payoff]
        if not myfilter: return dict()
        opt = [o for o,f in zip(self.option,myfilter) if f]
        mda = [m for m,f in zip(self.marketdata,myfilter) if f]
        idn = [i for i,f in zip(self.id_number,myfilter) if f]
        pricer = ClosedFormOptionPricerObject(
            payoff_type=payoff_type, 
            options = opt, 
            marketdata=mda, 
            id_number=idn, 
            sensitivities=self.sensitivities)
        return dict(zip(idn, pricer.main()))
    
    def main(self) -> List[OptionValuationResult]: 
        output = dict()
        for p in list(PayoffClosedFrom): 
            output.update(self.pricing(p))
        output = dict(sorted(output.items()))
        output = list(output.values())
        return output




    



    

    
