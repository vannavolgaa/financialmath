import numpy as np 
from financialmath.pricing.option.schema import ImpliedOptionMarketData
from financialmath.pricing.option.pde.interface import PDEBlackScholesPricerObject
from financialmath.instruments.option import *
import matplotlib.pyplot as plt

S = 100 
K = 100 
r = 0.001
q = 0.11
sigma = 0.16
t = 0.1
M = 100 
N = 300 
dt = 1/N 
dx = sigma * np.sqrt(2*dt)
Bu = 99.9
Bd = 99.9

mda = ImpliedOptionMarketData(S=S,r=r,q=q,sigma=sigma,F=np.nan)

opt_spec = OptionSpecification(
    strike=K, 
    tenor=OptionTenor(expiry=t), 
    barrier_up=Bu, 
    barrier_down=Bd)

opt_payoff = OptionPayoff(
    option_type=OptionalityType.call,
    exercise=ExerciseType.european,
    barrier_type=BarrierType.down_and_in, 
    barrier_obervation=ObservationType.continuous
)

opt = Option(opt_spec,opt_payoff)

test = PDEBlackScholesPricerObject(
    opt,
    mda,
    sensitivities=True, 
    use_thread=True, 
    N=N, 
    M=M)
valuation = test.valuation()
valuation.price
gridobj = test.generate_grid_object()