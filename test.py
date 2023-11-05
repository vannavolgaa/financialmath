import numpy as np 
from financialmath.pricing.option.schema import ImpliedOptionMarketData
from financialmath.pricing.option.pde.interface2 import PDEBlackScholesPricerObject
from financialmath.instruments.option import *


S = 100 
K = 120 
r = 0.01 
q = 0.02 
sigma = 0.2 
t = 1
M = 200 
N = 400 
dt = 1/N 
dx = sigma * np.sqrt(2*dt)
Bu = 105
Bd = 105

mda = ImpliedOptionMarketData(S=S,r=r,q=q,sigma=sigma,F=np.nan)

opt_spec = OptionSpecification(
    strike=K, 
    tenor=OptionTenor(expiry=t), 
    barrier_up=Bu, 
    barrier_down=Bd)

opt_payoff = OptionPayoff(
    option_type=OptionalityType.put,
    exercise=ExerciseType.european,
    barrier_obervation=ObservationType.in_fine, 
    barrier_type=BarrierType.up_and_out
)
opt = Option(opt_spec,opt_payoff)

S_vector = np.linspace(0,200,200)
opt.payoff_object(S_vector).barrier_condition()
test = PDEBlackScholesPricerObject(opt,mda,sensitivities=False, use_thread=False)
test2 = test.valuation()
test2.price

