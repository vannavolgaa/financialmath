from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List
from enum import Enum
import math

class TimeIntervalDefinition(Enum): 
    one_second = timedelta(seconds=1) 
    thirty_second = timedelta(seconds=30)
    one_minute = timedelta(minutes=1)
    three_minutes = timedelta(minutes=3) 
    five_minutes = timedelta(minutes=5) 
    ten_minutes = timedelta(minutes=10) 
    fifteen_minutes = timedelta(minutes=15) 
    thirty_minutes = timedelta(minutes=30) 
    one_hour = timedelta(hours=1) 
    two_hours = timedelta(hours=2)  
    three_hours = timedelta(hours=3)  
    six_hours = timedelta(hours=6) 
    twelve_hours = timedelta(hours=12)  
    daily = timedelta(days=1)  
    weekly = timedelta(weeks=1)  
    monthly = timedelta(days=30)
    quarterly = timedelta(days=90)
    semi_annually = timedelta(days=180)
    annually = timedelta(days=360)


@dataclass
class DataPoint:
    time : datetime
    value : str or int or float

@dataclass
class TimeSerieDefinition: 
    data : List[DataPoint]
    interval : TimeIntervalDefinition

    def __post_init__(self): 
        t = [d.reference_time for d in self.data]
        v = [d.value for d in self.data]
        data_as_dict = {k:d for k,d in zip(t,v)}
        data_as_dict=dict(sorted(data_as_dict.items()))
        self.times = list(data_as_dict.keys())
        self.values = list(data_as_dict.values())




