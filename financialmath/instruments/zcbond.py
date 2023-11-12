from dataclasses import dataclass

@dataclass
class ZCBond: 
    expiry: float 

@dataclass
class MarketZCBondQuotes: 
    bid : float 
    ask : float 
    zc_bond : ZCBond

    def __post_init__(self): 
        self.mid = (self.bid+self.ask)/2