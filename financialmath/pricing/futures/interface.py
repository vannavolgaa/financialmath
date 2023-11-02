from dataclasses import dataclass
from typing import List
from financialmath.instruments.futures import Future, FutureType
from financialmath.pricing.futures.schema import (FutureInputData, 
FutureSensibility, FutureValuationResult, FutureValuationFunction)
from financialmath.pricing.futures.pricing import (FuturePricing, InterestRateFuturePricing, 
BondFuturePricing, FuturePremiumList)
from financialmath.quanttool import QuantTool


@dataclass
class FutureValuationObject: 
    inputdata: FutureInputData or List[FutureInputData]
    instruments: Future or List[Future]
    future_type : FutureType
    continuous : bool = True 
    sensitivities : bool = True 

    def __post_init__(self): 
        self.instruments = QuantTool.convert_any_to_list(self.instruments)
        self.inputdata = QuantTool.convert_any_to_list(self.inputdata)
        self.n = len(self.instruments)
        self.valuation = self.valuation_object()
    
    def valuation_object(self) -> FutureValuationFunction: 
        match self.future_type: 
            case FutureType.bond:
                return BondFuturePricing(len(self.instruments))  
            case FutureType.interest_rate: 
                r =  [i.interest_rate_domestic for i in self.inputdata]
                return InterestRateFuturePricing(r=r) 
            case _: 
                t = [d.expiry for d in self.instruments]
                S = [i.spot for i in self.inputdata]
                x = self.premium()
                cont = self.continuous
                return FuturePricing(S=S, x=x, t=t,continuous=cont)

    def premium(self) -> FuturePremiumList: 
        ftype = self.future_type
        premium_obj = FuturePremiumList.get_premium_method(future_type=ftype)
        return [premium_obj(i) for i in self.inputdata]
    
    def main(self) -> List[FutureValuationResult]: 
        instruments = self.instruments
        marketdata = self.inputdata
        price = self.valuation.get_price(n=self.n)
        if self.sensitivities:
            sensi = self.valuation.get_all_sensisibilities(n=self.n)
        else: sensi = [FutureSensibility() for i in range(0, self.n)]
        method = self.valuation.get_method(n=self.n)
        data = zip(instruments, marketdata, price, sensi, method)
        return [FutureValuationResult(i,d,p,s,m) for i,d,p,s,m in data]
    
@dataclass
class FutureValuation: 

    inputdata: FutureInputData or List[FutureInputData]
    instruments: Future or List[Future]
    continuous : bool = True 
    sensitivities : bool = True 

    def __post_init__(self): 
        if not isinstance(self.instruments, list) : 
            self.instruments = [self.instruments]
            self.inputdata = [self.inputdata]
        self.id_number = list(range(0, self.n))
        self.future_type =  [f.future_type for f in self.instruments]
        self.expiry = [d.expiry for d in self.instruments]
    
    def pricing(self, fut_type : FutureType) -> dict: 
        myfilter = [(f == fut_type) for f in self.future_type]
        if not myfilter: return dict()
        fut = [o for o,f in zip(self.instruments,myfilter) if f]
        mda = [m for m,f in zip(self.inputdata,myfilter) if f]
        idn = [i for i,f in zip(self.id_number,myfilter) if f]
        pricer = FutureValuationObject(
            input_data=idn, 
            instruments = fut, 
            marketdata=mda, 
            id_number=idn, 
            sensitivities=self.sensitivities)
        return dict(zip(idn, pricer.main()))

    def main(self) -> List[FutureValuationResult]: 
        output = dict()
        for f in list(self.future_type): 
            output.update(self.pricing(f))
        output = dict(sorted(output.items()))
        output = list(output.values())
        return output



