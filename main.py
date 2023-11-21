from financialmath.model.blackscholes.montecarlo import MonteCarloBlackScholes, MonteCarloBlackScholesInput, BlackScholesDiscretization, RandomGeneratorType
import matplotlib.pyplot as plt
import numpy as np
import time 

mcinput = MonteCarloBlackScholesInput(

    S = 100,
    r=0.01, 
    q=0.02, 
    sigma = 0.2, 
    t=1,
    number_steps=1000,
    number_paths= 10000,
    discretization = BlackScholesDiscretization.milstein, 
    randoms_generator=RandomGeneratorType.quasiMC_halton
    
)

start = time.time()
test = MonteCarloBlackScholes(inputdata=mcinput).simulation()
end = time.time()
print(end-start)
plt.plot(np.transpose(test.initial))
plt.show()

