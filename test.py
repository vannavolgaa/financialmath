from financialmath.tools.date import * 
from datetime import datetime

from_date = datetime(year = 2021, month = 1, day = 1)
to_date = datetime.now()

convention_list = list(DayCountConvention)
output = dict()
for c in convention_list: 
    f = DayCountFactor(from_date,to_date,c).get()
    output[c.value] = f
print((output))