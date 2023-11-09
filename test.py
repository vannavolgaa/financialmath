from financialmath.tools.probability import ProbabilityDistribution
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from financialmath.tools.date import DayCountFactor, DayCountConvention

d1 = datetime.now()
d2 = datetime(year=2022,month=11,day=9)

for i in list(DayCountConvention):
    print(i.value)
    print(DayCountFactor(d2,d1,i).get())

