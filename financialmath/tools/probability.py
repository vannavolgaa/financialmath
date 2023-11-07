from dataclasses import dataclass, field
from enum import Enum
import scipy.stats 
import numpy as np
import sstudentt
from abc import abstractmethod, ABC

@dataclass
class DistributionMoment: 
    mean : float 
    standard_deviation : float 
    kewness : float 
    excess_kurtosis : float 


class ProbilityDistributionAbstract(ABC): 

    @abstractmethod
    def random(self, N:int, mu:float=0, sigma:float=1) -> np.array: pass

    @abstractmethod
    def pdf(self, x:int, mu:float=0, sigma:float=1) -> np.array: pass

    @abstractmethod
    def cdf(self, x:int, mu:float=0, sigma:float=1) -> np.array: pass

    @abstractmethod
    def moment(self, mu:float=0, sigma:float=1) -> DistributionMoment: pass

    def loglikelihood(self, x: float or np.array, 
                      mu:float=0, sigma:float=1) -> float: 
        return -np.log(self.pdf(x, mu=mu, sigma=sigma)).sum()
    
    def random_matrix(self, N: int, M:int, 
                      mu:float=0, sigma:float=1) -> np.array: 
        randoms = self.random(N = N*M, mu=mu, sigma=sigma)
        matrix_shape = (M,N)
        return np.reshape(randoms, matrix_shape)
    
    def correlated_random_matrix(self, corr_matrix: np.array, 
                                 N:int, M:int, 
                                 mu:float=0, sigma:float=1) -> np.array:
        random_matrix = self.random_matrix(M=M,N=N,mu=mu, sigma=sigma)
        cholesky_matrix = scipy.linalg.cholesky(corr_matrix).transpose()
        return cholesky_matrix.dot(random_matrix)
        
@dataclass
class NormalDistribution(ProbilityDistributionAbstract):

    theta : dict = field(default_factory=dict)
    distribution = scipy.stats.norm

    def random(self,N : int, 
               mu:float=0, sigma:float=1) -> np.array: 
        return self.distribution.rvs(size=N, loc = mu, scale = sigma)
    
    def pdf(self, x : float or np.array, 
            mu:float=0, sigma:float=1) -> np.array:
        return self.distribution.pdf(loc = mu, scale = sigma, x = x)
    
    def cdf(self, x : float or np.array,
            mu:float=0, sigma:float=1) -> np.array:
        return self.distribution.cdf(loc = mu, scale = sigma, x = x)
    
    def moment(self, mu:float=0, sigma:float=1) -> DistributionMoment: 
        return DistributionMoment(
            self.distribution.moment(order = 1, loc=mu, scale=sigma), 
            self.distribution.moment(order = 2, loc=mu, scale=sigma), 
            self.distribution.moment(order = 3, loc=mu, scale=sigma), 
            self.distribution.moment(order = 4, loc=mu, scale=sigma)-3)

@dataclass
class StudentTDistribution(ProbilityDistributionAbstract):
    theta : dict[str, float] = field(default_factory={'df' : 30})
    distribution = scipy.stats.t

    def __post_init__(self): 
        self.df = self.theta['df']
     
    def random(self,N : int,mu:float=0, sigma:float=1) -> np.array: 
        return self.distribution.rvs(
                df= self.df, 
                loc= mu, 
                scale= sigma, 
                size=N)
    
    def pdf(self, x : float or np.array,mu:float=0, sigma:float=1) -> np.array:
        return self.distribution.pdf(
            df = self.df, 
            loc = mu, 
            scale = sigma, 
            x = x
        )
    
    def cdf(self, x : float or np.array,mu:float=0, sigma:float=1) -> np.array:
        return self.distribution.cdf(
            df = self.df, 
            loc = mu, 
            scale = sigma, 
            x = x
        )

    def moment(self,mu:float=0, sigma:float=1) -> DistributionMoment: 
        return DistributionMoment(
            self.distribution.moment(order = 1, df = self.df, 
                                     loc=mu, scale=sigma), 
            self.distribution.moment(order = 2, df = self.df, 
                                     loc=mu, scale=sigma), 
            self.distribution.moment(order = 3, df = self.df, 
                                     loc=mu, scale=sigma), 
            self.distribution.moment(order = 4, df = self.df, 
                                     loc=mu, scale=sigma)-3)

@dataclass
class SkewedStudentTDistribution(ProbilityDistributionAbstract): 
    theta : dict[str, float] = field(default_factory={'df' : 30, 'tau' : 1})

    def __post_init__(self): 
        self.df, self.tau = self.theta['df'], self.theta['tau']

    def random(self,N : int, mu:float=0, sigma:float=1) -> np.array: 
        dtb = sstudentt.SST(mu=mu, sigma=sigma, nu=self.tau, tau=self.df)
        return dtb.r(n=N)
    
    def pdf(self, x : float or np.array, mu:float=0, sigma:float=1)->np.array:
        dtb = sstudentt.SST(mu=mu, sigma=sigma, nu=self.tau, tau=self.df)
        return dtb.d(x=x)
    
    def cdf(self, x : float or np.array, mu:float=0, sigma:float=1)->np.array:
        dtb = sstudentt.SST(mu=mu, sigma=sigma, nu=self.tau, tau=self.df)
        return dtb.p(x)
    
    def moment(self, mu:float=0, sigma:float=1) -> DistributionMoment: 
        randoms = self.random(N=50000, mu=mu, sigma=sigma)
        skew = scipy.stats.moment(randoms, moment=3)
        kurtosis = scipy.stats.moment(randoms, moment=4)
        return DistributionMoment(mu, sigma, skew, kurtosis-3)

class ProbabilityDistribution: 
    normal = NormalDistribution
    student_t = StudentTDistribution
    skewed_student_t = SkewedStudentTDistribution


