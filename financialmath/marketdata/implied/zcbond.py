from dataclasses import dataclass
from typing import List 
import numpy as np 
from financialmath.instruments.zcbond import MarketZCBondQuotes
from financialmath.marketdata.schema import ImpliedDataQuotes, ImpliedDataTypes
from financialmath.tools.tool import MainTool

@dataclass
class ImpliedRatesZCBond: 

    bond_quotes : MarketZCBondQuotes or List[MarketZCBondQuotes]
    continuous: bool = True

    def __post_init__(self): 
        if not isinstance(self.bond_quotes, list): 
            self.bond_quotes = [self.bond_quotes]
        self.bids = np.array([b.bid for b in self.bond_quotes])
        self.asks = np.array([b.ask for b in self.bond_quotes])
        self.t = np.array([b.zc_bond.expiry for b in self.bond_quotes])

    def get_yield(self, p:float or np.array, t:float or np.array)->List[float]:
        if self.continuous: y= np.log(p)/-t
        else: y= (1/p)**(1/t) - 1
        return MainTool.convert_array_to_list(y)
    
    def main(self) -> List[ImpliedDataQuotes]: 
        ask_yields = self.get_yield(self.bids,self.t)
        bid_yields = self.get_yield(self.asks,self.t)
        data_type = ImpliedDataTypes.interest_rate
        data = zip(bid_yields, ask_yields, self.bond_quotes)
        return [ImpliedDataQuotes(b,a,data_type,q) for b,a,q in data]






