from financialmath.model.svi import StochasticVolatilityInspired
import matplotlib.pyplot as plt 
import numpy as np

test = StochasticVolatilityInspired(
    atm_variance=0.01742625, 
    atm_skew=-0.1752111, 
    slope_put_wing=0.6997381, 
    slope_call_wing=0.8564763, 
    min_variance=0.0116249, 
    t=1) 

k_vector = np.linspace(-1,1,100)

#plt.plot(k_vector, test.total_variance(k=k_vector))
#plt.show()

plt.plot(k_vector, test.implied_volatility(k=k_vector))
plt.plot(k_vector, test.local_volatility(k=k_vector))
plt.show()