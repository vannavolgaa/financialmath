from datetime import datetime, timedelta
from dataclasses import dataclass
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
            match self.date_format: 
            case TimestampType.timestamp_in_mseconds: 
                return datetime.fromtimestamp(round(self.date/1000))
            case TimestampType.timestamp_in_seconds: 
                return datetime.fromtimestamp(self.date)
            case _: 
                return datetime.strptime(self.date, self.date_format)
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
    def range_of_business_dates(from_date:datetime, to_date:datetime, holiday : dict) -> List[datetime]: 
        f, t = from_date, to_date
        dates_range = DateTool.range_of_dates(from_date=from_date, to_date=to_date)
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
        f = self.from_date
        t = self.to_date
        self.delta_days = (t-f).days
        self.range_of_dates = DateTool.range_of_dates(f, t)

    def get_act_act_icma(self) -> float: 
        return np.nan 
    
    def get_act_act_isda(self) -> float: 
        n_in_leap = DateTool.count_days_in_leap_year(self.range_of_dates)
        return (n_in_leap/366)+(self.delta_days-n_in_leap)/365
    
    def get_act_365_leap(self) -> float: 
        if self.frequence==1: 
            cond = [DateTool.is_date_in_leap_year(d) 
                    for d in [self.from_date,self.to_date]] 
            if sum(cond)>=1: n=366 
            else: n=365
        else: 
            if DateTool.is_date_in_leap_year(self.to_date): n=366
            else: n=365 
        return self.delta_days/n
    
    def get_act_act_afb(self) -> float: 
        return np.nan 
    
    def get_thirty_360_bond_basis(self) -> float: 
        return np.nan 
    
    def get_thirtyE_360(self) -> float: 
        return np.nan 

    def get_thirtyE_360_isda(self) -> float: 
        return np.nan 

    def get_thirty_360_us(self) -> float: 
        return np.nan 

    def compute_actual(self, factor:int) -> float: 
        return self.delta_days/factor
    
    def compute_thirty_360_method(self) -> float:
        Ydays = 360*(self.to_date.year-self.from_date.year)
        Mdays = 30*(self.to_date.month-self.from_date.month)
        d1 = self.from_date.day
        d2 = self.to_date.day
        is_d2_last_day_month = (self.last_day_of_month(self.to_date) == self.to_date)
        is_d2_in_february = (self.to_date.month != 2)
        is_d1_last_day_month = (self.last_day_of_month(self.from_date) == self.from_date)
        is_d1_in_february = (self.from_date.month != 2)
        match self.convention: 
            case DayCountConvention.thirty_360_bond_basis:
                d1 = min([d1,30])
                if d1>29: d2 = min([d2,30])
            case DayCountConvention.thirtyE_360:
                d1 = min([d1,30])
                d2 = min([d2,30])
            case DayCountConvention.thirtyE_360_isda:
                if is_d1_last_day_month: 
                    d1=30
                if is_d2_in_february and is_d2_last_day_month: 
                    d2=30
            case DayCountConvention.thirty_360_us:
                d1_last_day_february = (is_d1_last_day_month and is_d1_in_february)
                d2_last_day_february = (is_d2_last_day_month and is_d2_in_february)
                if d1_last_day_february and d2_last_day_february: d2=30
                if d1_last_day_february: d1 = 30
                d1 = min([d1,30])
                if d1 ==30: d2 = min([d2,30])
            case _: return np.nan 
        return Ydays+Mdays+(d2-d1)

    def get(self) -> float: 
        match self.convention: 
            case DayCountConvention.actual_360: 
                return self.compute_actual(factor=360)
            case DayCountConvention.actual_365: 
                return self.compute_actual(factor=365)
            case DayCountConvention.actual_364: 
                return self.compute_actual(factor=364)
            case DayCountConvention.one_for_one: 
                return self.compute_actual(factor=365.25)
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

    


