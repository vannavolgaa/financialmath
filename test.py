from financialmath.tools.simulation import RandomGenerator, RandomGeneratorType, NormalDistribution
import numpy as np 
import matplotlib.pyplot as plt 

S = 100 
r = 0.01
q = 0.02 
t = 1 
sigma = 0.2
N = 300
M = 1000 
dt = t/N

generator =  RandomGenerator(
            probability_distribution=NormalDistribution(), 
            generator_type=RandomGenerator.standard)
Z = generator.generate(N=M*N)
Z = np.reshape(Z, (M,N))

dt_vector = np.cumsum(np.repeat(dt,N))
drift = np.exp((r-q)*dt)
diffusion = np.exp(-.5*(sigma**2)*dt+sigma *np.sqrt(dt)*Z)
simu1 = S*drift*np.cumprod(diffusion, axis = 1)

plt.plot(np.transpose(simu1))
plt.show()

cum_N = np.cumsum(np.repeat(1, N))

progressive_mean = np.cumsum(simu1,axis=1)/cum_N
np.mean(simu1,axis=1)
np.min(simu1,axis=1)
np.max(simu1,axis=1)



from financialmath.tools.tool import MainTool

mylist = [{0:'a'}, {1: 'b'}]

out = {}

[out.update(i) for i in mylist]

