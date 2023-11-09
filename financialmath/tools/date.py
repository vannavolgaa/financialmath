from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from typing import List
import numpy as np
import calendar

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

class TimestampType: 
    timestamp_in_mseconds = 1 
    timestamp_in_seconds = 2 

class DayCountConvention(Enum):
    actual_360 = 'Act/360' 
    actual_365 = 'Act/365'   
    actual_364 = 'Act/364'   
    actual_actual_isda = 'Act/Act ISDA' 
    actual_actual_icma = 'Act/Act ICMA' 
    actual_365_leap = 'Act/365L' 
    actual_actual_afb = 'Act/Act AFB'  
    thirty_360_bond_basis = '30/360 Bond Basis'
    thirty_360_us = '30/360 US'
    thirtyE_360 = '30E/360'
    thirtyE_360_isda = '30E/360 ISDA'
    one_for_one = '1/1'

@dataclass
class DateTool: 

    @staticmethod
    def process_date_to_datetime(date:str or int, 
                                date_format:str or TimestampType) -> datetime: 
        try: 
            match date_format: 
                case TimestampType.timestamp_in_mseconds: 
                    return datetime.fromtimestamp(round(date/1000))
                case TimestampType.timestamp_in_seconds: 
                    return datetime.fromtimestamp(date)
                case _: 
                    return datetime.strptime(date, date_format)
        except Exception as e: return None
        
    @staticmethod
    def last_day_of_month(date:datetime) -> datetime: 
        next_month = date.replace(day=28) + timedelta(days=4)
        return next_month - timedelta(days=next_month.day)
    
    @staticmethod
    def range_of_dates(from_date:datetime, to_date:datetime) -> List[datetime]: 
        f, t = from_date, to_date
        return [f+timedelta(days=x) for x in range((t-f).days)]
    
    @staticmethod
    def range_of_business_dates(from_date:datetime, to_date:datetime, 
                                holiday : dict) -> List[datetime]: 
        f, t = from_date, to_date
        dates_range = DateTool.range_of_dates(
            from_date=from_date, 
            to_date=to_date)
        return [d for d in dates_range if d in holiday]
    
    @staticmethod
    def is_date_in_leap_year(date:datetime) -> bool: 
        return calendar.isleap(date.year)
    
    @staticmethod
    def count_days_in_leap_year(range_of_dates: List[datetime]) -> int:  
        return sum([DateTool.is_date_in_leap_year(d) for d in range_of_dates])

@dataclass
class DayCountFactor: 
    from_date : datetime
    to_date : datetime
    convention :  DayCountConvention
    frequence : float = np.nan

    def __post_init__(self): 
        self.day_to_milliseconds = 24*60*60*1000
        f = self.from_date
        t = self.to_date
        self.delta_days = (t-f).days
        self.delta_milliseconds = (t-f).days*self.day_to_milliseconds
        self.year_360 = 360*self.day_to_milliseconds
        self.year_365 = 365*self.day_to_milliseconds
        self.year_364 = 364*self.day_to_milliseconds
        self.year_366 = 366*self.day_to_milliseconds
        self.year_36525 = 365.25*self.day_to_milliseconds
        self.d1 = f.day
        self.d2 = t.day
        self.d1_last_day_month = DateTool.last_day_of_month(f)
        self.d2_last_day_month = DateTool.last_day_of_month(t)
        self.is_d2_last_day_month = (self.d2_last_day_month == self.d2)
        self.is_d1_last_day_month = (self.d1_last_day_month == self.d1)
        self.is_d2_in_february = (self.d2 == 2)
        self.is_d1_in_february = (self.d1 == 2)

    def get_act_act_icma(self) -> float: 
        return np.nan 
    
    def get_act_act_isda(self) -> float: 
        dtom = self.day_to_milliseconds
        range_of_dates = DateTool.range_of_dates(self.from_date, self.to_date)
        delta_leap = DateTool.count_days_in_leap_year(range_of_dates)
        delta_not_leap = self.delta_days-delta_leap
        delta_leap_microseconds = (delta_leap*dtom)
        delta_not_leap_microseconds = (delta_not_leap*dtom)
        fraction_366 =  delta_leap_microseconds/self.year_366
        fraction_365 =  delta_not_leap_microseconds/self.year_365
        return fraction_366+fraction_365
    
    def get_act_365_leap(self) -> float: 
        return np.nan
    
    def get_act_act_afb(self) -> float: 
        return np.nan 
    
    def get_factor_360_method(self, d1: int, d2:int) -> float: 
        Y = self.year_360*(self.to_date.year-self.from_date.year)
        milliseconds_30days = 30*self.day_to_milliseconds
        M = milliseconds_30days*(self.to_date.month-self.from_date.month)
        d1 = timedelta(d1).microseconds
        d2 = timedelta(d2).microseconds
        return (Y+M +(d2-d1))/self.year_360

    def get_thirty_360_bond_basis(self) -> float: 
        d1 = min([self.d1,30])
        d2 = self.d2
        if d1>29: d2 = min([d2,30])
        return self.get_factor_360_method(d1,d2)
    
    def get_thirtyE_360(self) -> float: 
        return self.get_factor_360_method(min([self.d1,30]),min([self.d2,30]))

    def get_thirtyE_360_isda(self) -> float: 
        d1 = self.d1
        d2 = self.d2
        if self.is_d1_last_day_month: d1=30
        if self.is_d2_in_february and self.is_d2_last_day_month: d2=30
        return self.get_factor_360_method(d1,d2)
        
    def get_thirty_360_us(self) -> float: 
        d1 = self.d1
        d2 = self.d2
        d1_last_day_february = (self.is_d1_in_february 
                                and self.is_d1_last_day_month)
        d2_last_day_february = (self.is_d2_in_february 
                                and self.is_d2_last_day_month)
        if d1_last_day_february and d2_last_day_february: d2=30
        if d1_last_day_february: d1 = 30
        d1 = min([d1,30])
        if d1 ==30: d2 = min([d2,30]) 
        return self.get_factor_360_method(d1,d2)
    
    def compute_actual(self, factor:int) -> float: 
        return self.delta_milliseconds/factor

    def get(self) -> float: 
        match self.convention: 
            case DayCountConvention.actual_360: 
                return self.compute_actual(factor=self.year_360)
            case DayCountConvention.actual_365: 
                return self.compute_actual(factor=self.year_365)
            case DayCountConvention.actual_364: 
                return self.compute_actual(factor=self.year_364)
            case DayCountConvention.one_for_one: 
                return self.compute_actual(factor=self.year_36525)
            case DayCountConvention.actual_actual_isda: 
                return self.get_act_act_isda()
            case DayCountConvention.actual_actual_icma: 
                return self.get_act_act_icma()
            case DayCountConvention.actual_365_leap: 
                return self.get_act_365_leap()
            case DayCountConvention.actual_actual_afb: 
                return self.get_act_act_afb()
            case DayCountConvention.thirty_360_bond_basis: 
                return self.get_thirty_360_bond_basis()
            case DayCountConvention.thirty_360_us: 
                return self.get_thirty_360_us()
            case DayCountConvention.thirtyE_360: 
                return self.get_thirtyE_360()
            case DayCountConvention.thirtyE_360_isda: 
                return self.get_thirtyE_360_isda()
            case _: return np.nan

    


