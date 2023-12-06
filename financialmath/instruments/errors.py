
class LookBackOptionError(Exception): 
    """Lookback option cannot be set with in fine observation type"""

class NegativeTenorError(Exception): 
    """A negative tenor cannot be set for an instrument"""

class ForwardStartTenorError(Exception): 
    """Intermediate tenors must be superior or equal than the forwart start tenor"""
    
class ExpiryTenorError(Exception): 
    """Intermediate tenors must be inferior or equal than the expiry tenor"""   
    
    
    