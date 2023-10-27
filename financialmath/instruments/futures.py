from dataclasses import dataclass
from enum import Enum 

class FutureType(Enum):
    fx = 1 
    equity = 2 
    commodity = 3
    bond = 4
    interest_rate = 5
    crypto = 6

class SettlementType(Enum): 
    cash = 1
    physical = 2

class Future: 
    expiry: float 
    future_type : FutureType
    settlement_type : SettlementType


