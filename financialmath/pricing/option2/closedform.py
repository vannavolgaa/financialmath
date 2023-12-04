from dataclasses import dataclass
from enum import Enum
from typing import List
from financialmath.instruments.option import (Option, OptionPayoff, OptionalityType, 
    ExerciseType)
from financialmath.model.blackscholes.closedform import *
from financialmath.tools.tool import MainTool
from financialmath.pricing.schemas import OptionGreeks, OptionValuationResult

@dataclass
class BlackScholesParameters:
    S : float 
    r : float 
    q : float 
    sigma : float 
    
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
class ClosedFormBlackScholesPricer: 
    ids : List[int]
    payoff_type : PayoffClosedFrom
    inputdata: BlackScholesInputData

    def __post_init__(self): 
        method = ValuationMethodClosedFormMapping.find_method(self.payoff_type)
        self.pricer = method(self.inputdata)

@dataclass
class ClosedFormBlackScholesValuation: 

    options : List[Option] or Option
    parameters: List[BlackScholesParameters] or BlackScholesParameters
    with_greeks : bool = True

    greek_name = ['delta', 'vega', 'gamma', 'rho', 'epsilon', 'theta', 
                  'vanna', 'volga', 'speed', 'charm', 'veta', 'vera', 
                  'zomma', 'ultima', 'color']

    def __post_init__(self): 
        if not isinstance(self.option, list):
            self.parameters = [self.parameters]
            self.option = [self.option]
        self.n = len(self.option)
        self.id_number = list(range(0, self.n))
        self.payoffs = [o.payoff for o in self.option]
        self.closed_form_payoff = [PayoffClosedFrom.find_payoff(payoff=p)
                                    for p in self.payoffs]
        self.pricers = [self.pricer(p) for p in list(PayoffClosedFrom)]

    def method(self) -> str: 
        return 'Black-Scholes closed form option pricing formula'
    
    def pricer(self, payoff_type: PayoffClosedFrom)\
        -> ClosedFormBlackScholesPricer: 
        pfilter = [(clp == payoff_type) for clp in self.closed_form_payoff]
        if not pfilter: return None
        opt = [o for o,f in zip(self.option,pfilter) if f]
        param = [m for m,f in zip(self.parameters,pfilter) if f]
        ids = [i for i,f in zip(self.id_number,pfilter) if f]
        inputdata = BlackScholesInputData(
            S = np.array([p.S for p in param]),
            sigma = np.array([p.sigma for p in param]),
            r = np.array([p.r for p in param]),
            q = np.array([p.q for p in param]),
            K = np.array([o.specification.strike for o in opt]),
            t = np.array([o.specification.tenor.expiry for o in opt]))
        return ClosedFormBlackScholesPricer(
            ids = ids, payoff_type=payoff_type, 
            inputdata = inputdata)
    
    @staticmethod
    def read_numpy_array(x:np.array): 
        try: return list(x)
        except TypeError: return x

    def compute_price(self, pricer:ClosedFormBlackScholesPricer)\
        ->dict[int,float]: 
        prices = self.read_numpy_array(pricer.price())
        ids = pricer.ids
        try: return {i:p for i, p in zip(ids,prices)}
        except TypeError: return {i:prices for i in ids}
    
    def compute_greeks(self, pricer:ClosedFormBlackScholesPricer)\
        ->dict[int,OptionGreeks]: 
        ids = pricer.ids
        if self.with_greeks:
            greeks = [pricer.delta(), pricer.vega(), pricer.gamma(), 
                      pricer.rho(), pricer.epsilon(), pricer.theta(), 
                      pricer.vanna(), pricer.volga(),  pricer.speed(), 
                      pricer.charm(), pricer.veta(), pricer.vera(), 
                      pricer.zomma(), pricer.ultima(), pricer.color()]
            data = {n:self.read_numpy_array(d) for n,d in zip(self.greek_name, greeks)}
            try: 
                data = MainTool.dictlist_to_listdict(data)
                return {i:OptionGreeks(**d) for i,d in zip(ids,data)}
            except TypeError: 
                return {i:OptionGreeks(**data) for i in ids}
        else: return {i:OptionGreeks() for i in ids}
        
    def price(self) -> dict[int,float]:
        output = dict()
        for p in self.pricers:
            output.update(self.compute_price(p))
        return output
    
    def greeks(self) -> dict[int,OptionGreeks]:
        output = dict()
        for p in self.pricers:
            output.update(self.compute_greeks(p))
        return output
    
    def valuation(self) ->OptionValuationResult or List[OptionValuationResult]:
        prices = self.price()
        greeks = self.greeks()
        output = []
        for i in self.id_number:
            if i in list(prices.keys()): 
                output.append(OptionValuationResult(
                    self.option[i], self.parameters[i], 
                    prices[i], greeks[i],
                    self.method(),time_taken=0)) 
            else: 
                output.append(OptionValuationResult(
                    self.option[i], self.parameters[i], 
                    np.nan, OptionGreeks(),
                    self.method(),time_taken=0)) 
        return output




    

