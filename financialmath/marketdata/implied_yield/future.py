from dataclasses import dataclass
from typing import List 
import numpy as np 
from financialmath.instruments.futures import MarketFutureQuotes
from financialmath.instruments.spot import MarketSpotQuotes
from financialmath.instruments.zcbond import MarketZCBondQuotes
from financialmath.marketdata.schema import ImpliedDataQuotes, ImpliedDataTypes
from financialmath.tools.tool import MainTool
from financialmath.marketdata.implied_yield.zcbond import ImpliedRatesZCBond

@dataclass
class ImpliedYieldFutureOneZCBond: 

    future_quotes : MarketFutureQuotes or List[MarketFutureQuotes]
    zcbond_quotes : MarketZCBondQuotes or List[MarketZCBondQuotes]
    spot_quotes : MarketSpotQuotes or List[MarketSpotQuotes]

    def __post_init__(self): 
        if not isinstance(self.future_quotes, list): 
            self.future_quotes = [self.future_quotes]
            self.spot_quotes = [self.spot_quotes]
            self.zcbond_quotes = [self.zcbond_quotes]
        self.spot_bids = np.array([s.bid for s in self.spot_quotes])
        self.spot_asks = np.array([s.ask for s in self.spot_quotes])
        self.future_bids = np.array([f.bid for f in self.future_quotes])
        self.future_asks = np.array([f.ask for f in self.future_quotes])
        self.t = np.array([f.future.expiry for f in self.future_quotes])
        self.implied_rates = ImpliedRatesZCBond(self.zcbond_quotes, 
                                                True).main()
        self.rate_bid = np.array([i.bid for i in self.implied_rates])
        self.rate_ask = np.array([i.ask for i in self.implied_rates])
    
    def get_yield(self,S:float or np.array, F:float or np.array, 
                  t: float or np.array, r:float or np.array)->List[float]:
        y = (np.log(F/S)/(r*t))/-t
        return MainTool.convert_array_to_list(y)
    
    def main(self) -> List[ImpliedDataQuotes]: 
        ask_yields = self.get_yield(self.spot_bids,self.future_asks, self.t, 
                                    self.rate_ask)
        bid_yields = self.get_yield(self.spot_asks,self.future_bids, self.t,
                                    self.rate_bid)
        data_type = ImpliedDataTypes.dividend_yield
        data = zip(bid_yields, ask_yields, 
                   self.spot_quotes, self.future_quotes, self.zcbond_quotes)
        return [ImpliedDataQuotes(b,a,data_type,[s,f,z]) for b,a,s,f,z in data]

@dataclass
class ImpliedYieldFutureTwoZCBond: 

    future_quotes : MarketFutureQuotes or List[MarketFutureQuotes]
    spot_quotes : MarketSpotQuotes or List[MarketSpotQuotes]
    domestic_zcbond_quotes : MarketZCBondQuotes or List[MarketZCBondQuotes]
    foreign_zcbond_quotes : MarketZCBondQuotes or List[MarketZCBondQuotes]

    def __post_init__(self): 
        if not isinstance(self.future_quotes, list): 
            self.future_quotes = [self.future_quotes]
            self.spot_quotes = [self.spot_quotes]
            self.domestic_zcbond_quotes = [self.domestic_zcbond_quotes]
            self.foreign_zcbond_quotes = [self.foreign_zcbond_quotes]
        self.spot_bids = np.array([s.bid for s in self.spot_quotes])
        self.spot_asks = np.array([s.ask for s in self.spot_quotes])
        self.future_bids = np.array([f.bid for f in self.future_quotes])
        self.future_asks = np.array([f.ask for f in self.future_quotes])
        self.t = np.array([f.future.expiry for f in self.future_quotes])
        self.dom_implied_rates=ImpliedRatesZCBond(self.domestic_zcbond_quotes, 
                                                    True).main()
        self.for_implied_rates=ImpliedRatesZCBond(self.foreign_zcbond_quotes, 
                                                    True).main()
        self.domrate_bid = [i.bid for i in self.dom_implied_rates]
        self.domrate_ask = [i.ask for i in self.dom_implied_rates]
        self.forrate_bid = [i.bid for i in self.for_implied_rates]
        self.forrate_ask = [i.ask for i in self.for_implied_rates]
    
    def get_yield(self,S:float or np.array, F:float or np.array, 
                  t: float or np.array, rd:float or np.array, 
                  rf:float or np.array)->List[float]:
        y = (np.log(F/S)/((rd-rf)*t))/t
        return MainTool.convert_array_to_list(y)
    
    def main(self) -> List[ImpliedDataQuotes]: 
        ask_yields = self.get_yield(self.spot_bids,self.future_asks, self.t, 
                                    self.domrate_ask, self.forrate_bid)
        bid_yields = self.get_yield(self.spot_asks,self.future_bids, self.t,
                                    self.domrate_bid, self.forrate_ask)
        data_type = ImpliedDataTypes.basis_spread
        data = zip(bid_yields, ask_yields, 
                   self.spot_quotes, self.future_quotes, 
                   self.domestic_zcbond_quotes, self.foreign_zcbond_quotes)
        return [ImpliedDataQuotes(b,a,data_type,[s,f,zd,zf]) 
                for b,a,s,f,zd, zf in data]

@dataclass
class ImpliedYieldFuture: 

    future_quotes : MarketFutureQuotes or List[MarketFutureQuotes]
    spot_quotes : MarketSpotQuotes or List[MarketSpotQuotes]
    continuous: bool = True

    def __post_init__(self): 
        if not isinstance(self.future_quotes, list): 
            self.future_quotes = [self.future_quotes]
            self.spot_quotes = [self.spot_quotes]
        self.spot_bids = np.array([s.bid for s in self.spot_quotes])
        self.spot_asks = np.array([s.ask for s in self.spot_quotes])
        self.future_bids = np.array([f.bid for f in self.future_quotes])
        self.future_asks = np.array([f.ask for f in self.future_quotes])
        self.t = np.array([f.future.expiry for f in self.future_quotes])

    def get_yield(self,S:float or np.array, F:float or np.array, 
                  t: float or np.array)->List[float]:
        if self.continuous: y = np.log(F/S)/t
        else : y = (F/S)**(1/t)-1
        return MainTool.convert_array_to_list(y)
    
    def main(self) -> List[ImpliedDataQuotes]: 
        ask_yields = self.get_yield(self.spot_bids,self.future_asks, self.t)
        bid_yields = self.get_yield(self.spot_asks,self.future_bids, self.t)
        data_type = ImpliedDataTypes.interest_rate
        data = zip(bid_yields, ask_yields, self.spot_quotes, self.future_quotes)
        return [ImpliedDataQuotes(b,a,data_type,[s,f]) for b,a,s,f in data]