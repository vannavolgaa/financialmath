from datetime import datetime 
from dataclasses import dataclass

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


@dataclass
class ProcessDate: 
    date : str or int 
    date_format : str or TimestampType

    def __post_init__(self): 
        try : self.processed_date = self.process_date_to_datetime()
        except Exception as e: 
            print('Date processing did not work.')
            self.processed_date = None

    def process_date_to_datetime(self) -> datetime: 
        match self.date_format: 
            case timestamp_in_mseconds: 
                return datetime.fromtimestamp(round(self.date/1000))
            case timestamp_in_seconds: return datetime.fromtimestamp(self.date)
            case _: return datetime.strptime(self.date, self.date_format)