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

class MainTool: 

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
        return [dict(zip(mydict,t)) for t in zip(*mydict.values())]
    
    @staticmethod
    def listdict_to_dictlist_samekey(mylist: List[dict]): 
        return {k: [dic[k] for dic in mylist] for k in mylist[0]}
    
    @staticmethod
    def listdict_to_dictlist_diffkey(mylist: List[dict]): 
        common_keys = set.intersection(*map(set, mylist))
        return {k: [dic[k] for dic in mylist] for k in common_keys}
    
    @staticmethod
    def listdict_to_dictlist(mylist: List[dict]): 
        out = {}
        [out.update(i) for i in mylist]
        return out 

    @staticmethod
    def convert_to_numpy_array(x:float or List[float]): 
        if isinstance(x, list): return np.array(x)
        else: return x
    
    @staticmethod
    def convert_array_to_list(x: float or np.array): 
        if isinstance(x, float): return [x]
        else: return list(x)
    
    @staticmethod
    def convert_any_to_list(x): 
        if not isinstance(x, list): return [x]
        else: return x

    @staticmethod
    def order_join_lists(keys:List, values:List, 
                                by_key:bool = True, reverse = False): 
        data = dict(zip(keys,values))
        if by_key: return dict(sorted(data.items(), reverse=reverse))
        else: return sorted(data.items(), key=lambda x: x[1], reverse=reverse)   



    





