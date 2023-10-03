from financialmath.instruments.option import CreateOption
from financialmath.pricing.option.obj import ImpliedOptionMarketData
from financialmath.pricing.option.closedform.interface import ClosedFormOptionPricer

import numpy as np
K = 100*np.linspace(0.5,1.5,100)
t = np.linspace(0.1,10,100)
calls = [CreateOption.european_vanilla_call(strike=KK, expiry=tt) for KK, tt in zip(K,t)]
#put = CreateOption.european_vanilla_put(strike=100, expiry=1)
mda = [ImpliedOptionMarketData(S=100, r=0.01, q=0.02, sigma = 0.2, F = None) for i in range(0,100)]

cfop = ClosedFormOptionPricer(marketdata=mda, option=calls)

test = cfop.main()

opt0 = test[0]
opt0.price
opt0.sensitivities
