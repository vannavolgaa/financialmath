from financialmath.marketdata.implied.future import *
from financialmath.instruments.futures import (Future, MarketFutureQuotes, 
                                               FutureType, SettlementType)
from financialmath.instruments.spot import Spot, MarketSpotQuotes
from financialmath.instruments.zcbond import ZCBond, MarketZCBondQuotes

spot = Spot()
future = Future(1,FutureType.equity,SettlementType.cash)
zcbond = ZCBond(expiry=1)
future_quote= MarketFutureQuotes(115,125,future)
spot_quote= MarketSpotQuotes(105,108,spot)
bond_quote = MarketZCBondQuotes(85,89,zcbond)

ImpliedDividendEquityFutures(future_quote, bond_quote,spot_quote, False).main()
ImpliedDividendEquityFutures(future_quote, bond_quote,spot_quote, True).main()





