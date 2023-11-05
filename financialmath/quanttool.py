import numpy as np 
from typing import List
import threading 
import sys as _sys

class ThreadWithReturn(threading.Thread):

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, *, daemon=None):
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self.__value = None

    def run(self):
        try:
            if self._target:
                return self._target(*self._args, **self._kwargs)
        finally:
            del self._target, self._args, self._kwargs

    def _bootstrap_inner(self):
        try:
            self._set_ident()
            self._set_tstate_lock()
            self._started.set()
            with threading._active_limbo_lock:
                threading._active[self._ident] = self
                del threading._limbo[self]

            if threading._trace_hook:
                _sys.settrace(threading._trace_hook)
            if threading._profile_hook:
                threading. _sys.setprofile(threading._profile_hook)

            try:
                self.__value = True, self.run()
            except SystemExit:
                pass
            except:
                exc_type, exc_value, exc_tb = self._exc_info()
                self.__value = False, exc_value
                if _sys and _sys.stderr is not None:
                    print("Exception in thread %s:\n%s" %
                          (self.name, threading._format_exc()), file=_sys.stderr)
                elif self._stderr is not None:
                    try:
                        print((
                            "Exception in thread " + self.name +
                            " (most likely raised during interpreter shutdown):"), file=self._stderr)
                        print((
                            "Traceback (most recent call last):"), file=self._stderr)
                        while exc_tb:
                            print((
                                '  File "%s", line %s, in %s' %
                                (exc_tb.tb_frame.f_code.co_filename,
                                    exc_tb.tb_lineno,
                                    exc_tb.tb_frame.f_code.co_name)), file=self._stderr)
                            exc_tb = exc_tb.tb_next
                        print(("%s: %s" % (exc_type, exc_value)), file=self._stderr)
                    finally:
                        del exc_type, exc_value, exc_tb
            finally:
                pass
        finally:
            with threading._active_limbo_lock:
                try:
                    del threading._active[threading.get_ident()]
                except:
                    pass

    @property
    def returned(self):
        if self.__value is None:
            self.join()
        if self.__value is not None:
            valid, value = self.__value
            if valid:
                return value
            raise value

class QuantTool: 

    @staticmethod
    def send_tasks_with_threading(task : object, args: List[tuple]) -> List:
        output = []
        for a in args:
            threads = []
            thread = ThreadWithReturn(target=task,
                                    args=a, daemon=True)
            thread.start()
            threads.append(thread)
            output.append([thread.returned for thread in threads])
        return output 

    @staticmethod
    def dictlist_to_listdict(mydict: dict): 
        keys = list(mydict.keys())
        length = len(mydict[keys[0]])
        output = []
        for i in range(0, length): 
            output.append({k : mydict[k][i] for k in keys})
        return output 

    
    @staticmethod
    def convert_to_numpy_array(x:float or List[float]): 
        if isinstance(x, list): return np.array(x)
        else: return x
    
    @staticmethod
    def compute_blackscholes_d1(S, K, r, q, t, sigma):
        return(np.log(S/K)+(r-q+sigma**2/2)*t)/(sigma*np.sqrt(t))
    
    @staticmethod
    def compute_blackscholes_d2(S, K, r, q, t, sigma):
        d1 = QuantTool.compute_blackscholes_d1(S=S,K=K,t=t,r=r,q=q, sigma=sigma)
        return d1-sigma*np.sqrt(t)
    
    @staticmethod
    def convert_array_to_list(x: float or np.array): 
        if isinstance(x, float): return [x]
        else: return list(x)
    
    @staticmethod
    def convert_any_to_list(x): 
        if not isinstance(x, list): return [x]
        else: return x

    





