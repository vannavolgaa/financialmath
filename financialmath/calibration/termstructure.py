from dataclasses import dataclass
from typing import List
import numpy as np 
from financialmath.instruments.option import (
    MarketOptionQuotes, 
    OptionalityType
    )
from financialmath.instruments.spot import MarketSpotQuotes
from financialmath.instruments.futures import MarketFutureQuotes
from financialmath.marketdata.termstructure import (
    YieldCurve, 
    TermStructure, 
    ExtrapolatedTermStructure
    )

@dataclass
class BidAskImpliedRates: 
    bid : np.array 
    ask : np.array
    t : np.array

    def __post_init__(self): 
        self.mid = (self.bid+self.ask)/2
        self.unique_t = list(set(self.t))
    
    def average_yield_by_tenor(self) -> np.array: 
        yields = []
        for t in self.unique_t: 
            tpos = np.where(self.t==t)
            iy = self.mid[tpos]
            yields.append(np.mean(iy))
        return yields
    
    def extrapolated_term_structure(self) -> ExtrapolatedTermStructure:
        if len(self.unique_t)==len(self.t): 
            return ExtrapolatedTermStructure(
                t = self.t, 
                yields = self.mid
            ) 
        else: 
            return ExtrapolatedTermStructure(
                t = self.t, 
                yields = self.average_yield_by_tenor()
            ) 

@dataclass
class ImpliedRatesCalibrationFutures: 
    future_quotes : List[MarketFutureQuotes]
    spot_quote : MarketSpotQuotes
    yield_curve : YieldCurve or TermStructure = None

    def __post_init__(self): 
        self.fut_t = [f.future.expiry for f in self.future_quotes]
        self.fut_bid = np.array([f.bid for f in self.future_quotes])
        self.fut_ask = np.array([f.ask for f in self.future_quotes])
        self.spot_bid, self.spot_ask = self.spot_quote.bid, self.spot_quote.ask
    
    def discount_rate(self) -> BidAskImpliedRates: 
        return BidAskImpliedRates(
            bid = np.log(self.fut_ask/self.spot_bid)/self.fut_t,
            ask = np.log(self.fut_bid/self.spot_ask)/self.fut_t,
            t = self.fut_t
        )
    
    def carry_cost(self) -> BidAskImpliedRates: 
        dr = self.yield_curve.rate(t=self.fut_t)
        return BidAskImpliedRates(
            bid = dr-np.log(self.fut_bid/self.spot_ask)/self.fut_t,
            ask = dr-np.log(self.fut_ask/self.spot_bid)/self.fut_t,
            t = self.fut_t
        )

@dataclass
class ImpliedRatesCalibrationEuropeanOptions: 
    option_quotes : List[MarketOptionQuotes]
    spot_quote : MarketSpotQuotes
    yield_curve : YieldCurve or TermStructure = None

    def __post_init__(self): 
        self.opt_t = [o.specification.tenor.expiry for o in self.options]
        self.pcp_instruments = self.putcall_parity_instruments()
        self.spot_bid, self.spot_ask = self.spot_quote.bid, self.spot_quote.ask
        self.puts = self.pcp_instruments['puts']
        self.calls = self.pcp_instruments['calls']
        self.t = np.array([p.option.specification.tenor.expiry for p in self.puts])
        self.K = np.array([p.option.specification.strike for p in self.puts])
        self.put_bids = np.array([p.bid for p in self.puts])
        self.put_asks = np.array([p.ask for p in self.puts])
        self.call_bids = np.array([c.bid for c in self.calls])
        self.call_asks = np.array([c.ask for c in self.calls])
        
    def putcall_parity_instruments(self) -> dict[str,List[MarketOptionQuotes]]:
        puts = [q for q in self.option_quotes 
                if q.option.payoff.option_type == OptionalityType.put]
        calls = [q for q in self.option_quotes 
                if q.option.payoff.option_type == OptionalityType.call]
        strike_cond = [(p.option.specification.strike == \
                        c.option.specification.strike) 
                      for p,c in zip(puts, calls)]
        tenor_cond = [(p.option.specification.tenor.expiry == \
                       c.option.specification.tenor.expiry)
                      for p,c in zip(puts, calls)]
        #future_cond = [(p.option.payoff.future == c.option.payoff.future) 
                       #for p,c in zip(puts, calls)]
        cond = (strike_cond and tenor_cond)
        return {'puts': [p for p,cd in zip(puts, cond) if cd], 
                'calls': [c for c,cd in zip(calls, cond) if cd]}
    
    def discount_rate(self) -> BidAskImpliedRates: 
        ca, cb = self.call_asks,self.call_bids
        pa, pb = self.put_asks,self.put_bids
        sa, sb = self.spot_ask, self.spot_bid
        K, t = self.K, self.t
        return BidAskImpliedRates(
            bid = np.log((sa+pa-cb)/K)/-t, 
            ask = np.log((sb+pb-ca)/K)/-t,
            t = t
        )
    
    def carry_cost_future(self) -> BidAskImpliedRates: 
        ca, cb = self.call_asks,self.call_bids
        pa, pb = self.put_asks,self.put_bids
        sa, sb = self.spot_ask, self.spot_bid
        K, t = self.K, self.t
        dr = self.yield_curve.rate(t=t)
        return BidAskImpliedRates(
            bid = np.log((ca - pb + np.exp(-dr*t)*K)/sb)/-t, 
            ask = np.log((cb - pa + np.exp(-dr*t)*K)/sa)/-t, 
            t = t
        )




