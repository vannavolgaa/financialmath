from dataclasses import dataclass
from typing import List
from enum import Enum
import math
from datetime import datetime
from financialmath.tools.date import (ProcessDate, TimeIntervalDefinition, 
TimestampType)
from financialmath.tools.tool import MainTool

@dataclass
class DataPoint:
    time : datetime
    value : str or int or float
    log_difference : float 
    absolute_difference : float 
    relative_difference : float 

@dataclass
class TimeSerieObject: 
    data : List[DataPoint]
    interval : TimeIntervalDefinition

    def __post_init__(self): 
        pass

@dataclass
class ProcessTimeSerie: 
    dates : List[str or int]
    values : List[float]
    interval : TimeIntervalDefinition
    date_format : str or TimestampType

    def __post_init__(self): 
        self.n = len(self.dates)
        dates = self.process_dates()
        order_dict = MainTool.order_join_lists(keys=dates, values=values)
        self.dates, self.values = list(order_dict.keys()), list(order_dict.values())
        self.log_diff = self.compute_log_return()
        self.absolute_diff = self.compute_simple_difference(self.values)
        self.relative_diff = self.compute_simple_return(self.values)

    def process_dates(self) -> List[datetime]: 
        return [ProcessDate(date=d, date_format=self.date_format).processed_date 
                for d in self.dates]

    def compute_log_return(self) -> np.array: 
        try:return np.diff(np.log(self.values), prepend=np.nan)
        except Exception as e: return np.repeat(np.nan, len(self.values))

    def compute_simple_difference(self) -> np.array: 
        try:return np.diff(self.values, prepend=np.nan)
        except Exception as e: return np.repeat(np.nan, len(self.values))

    def compute_simple_return(self) -> np.array: 
        try:return ProcessTimeSerie.compute_simple_difference()/self.values
        except Exception as e: return np.repeat(np.nan, len(self.values))
    
    def process_to_object(self) -> TimeSerieObject: 
        datapoints = [DataPoint(d, v, lg, ad, rd) 
                        for d,v,lg,ad,rd in zip(self.dates,self.values, 
                        self.log_diff, self.absolute_diff, self.relative_diff)]
        return TimeSerieObject(data=datapoints, interval=self.interval)

        






