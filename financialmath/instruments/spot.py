from dataclasses import dataclass

@dataclass
class Spot: 
    pass

@dataclass
class MarketSpotQuotes: 
    bid : float 
    ask : float 
    spot : Spot

    def __post_init__(self): 
        self.mid = (self.bid+self.ask)/2