from dataclasses import dataclass
from enum import Enum
from typing import List
import numpy as np
from financialmath.pricing.option.closedform.framework.blackscholes import *
from financialmath.instruments.option import (Option, OptionPayoff, OptionalityType, 
    ExerciseType)
from financialmath.pricing.option.schema import (ImpliedOptionMarketData, OptionValuationFunction, 
                                              OptionGreeks, OptionValuationResult)


class PayoffClosedFrom(Enum): 
    unknown = 1
    european_vanilla_call = OptionPayoff(option_type=OptionalityType.call,
                            exercise=ExerciseType.european) 
    european_vanilla_put = OptionPayoff(option_type=OptionalityType.put, 
                            exercise=ExerciseType.european)   
    fut_european_vanilla_call = OptionPayoff(option_type=OptionalityType.call,
                            exercise=ExerciseType.european, future=True) 
    fut_european_vanilla_put = OptionPayoff(option_type=OptionalityType.put, 
                            exercise=ExerciseType.european, future=True)

    @staticmethod
    def find_payoff(payoff : OptionPayoff): 
        try : return [pcf for pcf in list(PayoffClosedFrom) 
                        if pcf.value == payoff][0]
        except Exception as e: return PayoffClosedFrom.unknown

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
    id_number : List[int]
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
        self.valuationmethod = ValuationMethodClosedFormMapping.find_method(self.payoff_type)
        self.get_model()

    def get_model(self): 
        try: self.model = self.valuationmethod(inputdata=self.inputdata)
        except Exception as e: self.model = None

    def compute_greeks(self) -> List[OptionGreeks]: 
        fall_back_greeks = [OptionGreeks() for i in range(0,self.n)]
        try: 
            if self.sensitivities:
                delta = QuantTool.convert_array_to_list(self.model.delta())
                vega = QuantTool.convert_array_to_list(self.model.vega())
                theta = QuantTool.convert_array_to_list(self.model.theta())
                rho = QuantTool.convert_array_to_list(self.model.rho())
                epsilon = QuantTool.convert_array_to_list(self.model.epsilon())
                gamma = QuantTool.convert_array_to_list(self.model.gamma())
                vanna = QuantTool.convert_array_to_list(self.model.vanna())
                volga = QuantTool.convert_array_to_list(self.model.volga())
                ultima = QuantTool.convert_array_to_list(self.model.ultima())
                speed = QuantTool.convert_array_to_list(self.model.speed())
                zomma = QuantTool.convert_array_to_list(self.model.zomma())
                color = QuantTool.convert_array_to_list(self.model.color())
                veta = QuantTool.convert_array_to_list(self.model.veta())
                #vera = QuantTool.convert_array_to_list(self.model.vera())
                charm = QuantTool.convert_array_to_list(self.model.charm())
         
                return [OptionGreeks(delta=delta[i], vega=vega[i], 
                                    theta=theta[i], rho=rho[i], 
                                    epsilon=epsilon[i], gamma=gamma[i], 
                                    vanna=vanna[i], volga=volga[i], 
                                    charm=charm[i], veta=veta[i], 
                                    vera=np.nan, speed=speed[i], 
                                    zomma=zomma[i], color=color[i], 
                                    ultima=ultima[i]) for i in range(0,self.n)]
            else: return fall_back_greeks
        except Exception as e: 
            print(str(e))
            return fall_back_greeks

    def compute_price(self) -> List[float]: 
        try: 
            return QuantTool.convert_array_to_list(self.model.price())
        except Exception as e: 
            return [np.nan for i in range(0,self.n)]

    def get_method_name(self) -> str: 
        try: return self.model.method()
        except Exception as e: return None 
    
    def send_to_valuation_schema(self):
        
        return [OptionValuationResult(instrument=i, price=p, 
                                        sensitivities=g, 
                                        method = self.get_method_name(), 
                                        marketdata=m) 
                                        for i, p, g, m in 
                                        zip(
                                        self.options, 
                                        self.compute_price(), 
                                        self.compute_greeks(), 
                                        self.marketdata
                                        )
                ]

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
    
    def pricing(self, payoff_type: PayoffClosedFrom): 
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
        return dict(zip(idn, pricer.send_to_valuation_schema()))
    
    def main(self): 
        output = dict()
        for p in list(PayoffClosedFrom): 
            output.update(self.pricing(p))
        output = dict(sorted(output.items()))
        output = list(output.values())
        if len(output)==1: return output[0]
        else: return output




    



    

    
