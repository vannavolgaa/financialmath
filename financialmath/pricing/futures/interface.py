from dataclasses import dataclass
from financialmath.instruments.futures import Future, FutureType
from financialmath.pricing.futures.schema import (FutureInputData, 
FutureSensibility, FutureValuationResult)
from financialmath.pricing.futures.pricing import (FuturePricing, InterestRateFuturePricing, 
BondFuturePricing, FuturePremiumList)


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
        self.valuation = self.valuationobject()
    
    def valuationobject(self): 
        match self.future_type: 
            case FutureType.bond:
                return InterestRateFuturePricing(r=r) 
            case FutureType.interest_rate: 
                r =  [i.interest_rate_domestic for i in self.inputdata]
                return InterestRateFuturePricing(r=r)
            case _: 
                expiry = [f.expiry for d in self.instruments]
                spot = [i.spot for i in self.inputdata]
                x = self.premium()
                return FuturePricing(S=spot, x=x, t=expiry, 
                                    continuous=self.continuous)

    def premium(self): 
        premiumobject = FuturePremiumList.get_premium_method(
                                future_type=self.future_type)
        return [premiumobject(i) for i in self.inputdata]
    
    def price(self): 
        return QuantTool.convert_array_to_list(self.valuation.price())
    
    def sensitivities(self): 
        if self.sensitivities:
            delta = QuantTool.convert_array_to_list(self.valuation.delta())
            rho = QuantTool.convert_array_to_list(self.valuation.rho())
            theta = QuantTool.convert_array_to_list(self.valuation.theta())
            return [FutureSensibility(
                                    delta=delta[i],
                                    theta=theta[i], 
                                    rho=rho[i]
                                    ) for i in range(0,self.n)]
        else: 
            return [FutureSensibility() for i in range(0, self.n)]
    
    def method(self): 
        return [self.valuation.method()]*self.n 
    
    def send_to_valuation_schema(self): 
        return FutureValuationResult(instrument=self.instruments, 
                                    marketdata=self.inputdata, 
                                    price=self.price(), 
                                    sensitivities=self.sensitivities(), 
                                    method= self.method())    

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
        self.expiry = [f.expiry for d in self.instruments]
    
    def pricing(self, fut_type : FutureType): 
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
        return dict(zip(idn, pricer.send_to_valuation_schema()))

    def main(self): 
        output = dict()
        for f in list(self.future_type): 
            output.update(self.pricing(f))
        output = dict(sorted(output.items()))
        output = list(output.values())
        if len(output)==1: return output[0]
        else: return output



