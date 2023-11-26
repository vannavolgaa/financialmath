from dataclasses import dataclass 
from enum import Enum 
import numpy as np
from scipy.stats.qmc import Sobol, Halton
import scipy.linalg 
from financialmath.tools.probability import (ProbilityDistributionAbstract, 
NormalDistribution, UniformDistribution)
import scipy.stats 

class RandomGeneratorType: 
    standard = 1 
    antithetic = 2 
    quasiMC_sobol = 3 
    quasiMC_halton = 4

@dataclass
class RandomGenerator:
    mu : float = 0
    sigma : float = 1 
    probability_distribution: ProbilityDistributionAbstract = NormalDistribution()
    generator_type : RandomGeneratorType = RandomGeneratorType.standard
    
    def standard(self, N:int) -> np.array: 
        return self.probability_distribution.random(
            N= N, mu = self.mu, sigma=self.sigma)

    def antithetic(self, N:int) -> np.array:
        u = UniformDistribution().random(N = round(N/2))
        u = np.concatenate((u,1-u))
        z = self.probability_distribution.inverse_cdf(
            x = u, mu = self.mu, sigma=self.sigma)
        return z
    
    def quasiMC_from_sobol(self,N:int) -> np.array:
        d = round(np.log(N)/np.log(2))
        sobol_seq = Sobol(d=1).random_base2(d)
        sobol_seq = np.reshape(sobol_seq, (1,len(sobol_seq)))
        sobol_seq = sobol_seq[0]
        randoms = self.probability_distribution.inverse_cdf(
                    x = sobol_seq, mu = self.mu, sigma = self.sigma)
        return randoms
    
    def quasiMC_from_halton(self,N:int) -> np.array:
        halton_seq = np.reshape(Halton(d=1).random(N), (1,N))
        halton_seq = halton_seq[0]
        randoms = self.probability_distribution.inverse_cdf(
                    x = halton_seq, mu = self.mu, sigma = self.sigma)
        return randoms
    
    def generate(self, N: int) -> np.array: 
        match self.generator_type: 
            case RandomGeneratorType.standard: 
                return self.standard(N) 
            case RandomGenerator.antithetic: 
                return self.antithetic(N)
            case RandomGenerator.quasiMC_from_halton: 
                return self.quasiMC_from_halton(N)
            case RandomGenerator.quasiMC_from_sobol: 
                return self.quasiMC_from_sobol(N)
            case _: return self.standard(N)
    
    def generate_correlated(self, N : int, correlation_matrix:np.array) -> np.array: 
        cholesky_matrix = scipy.linalg.cholesky(correlation_matrix).transpose()
        M = correlation_matrix.shape[0]
        randoms = self.generate(N*M)
        randoms = np.reshape(randoms, (M,N))
        return cholesky_matrix.dot(randoms)



